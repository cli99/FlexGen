"""
Run OPT with huggingface or deepspeed.

Usage:
deepspeed --num_gpus 1 hf_opt.py --model facebook/opt-1.3b --batch-size 16 --use-deepspeed --cpu-offload

Reference:
https://github.com/huggingface/transformers-bloom-inference/tree/main/bloom-inference-scripts
"""

import argparse
import multiprocessing as mp
import os
import pickle
import time
import gc

import numpy as np

from accelerate import (infer_auto_device_map, init_empty_weights,
    load_checkpoint_and_dispatch)
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from models.modeling_opt import OPTForCausalLM
import torch

from timer import timers
from utils import (disable_torch_init, GB, project_decode_latency,
    write_benchmark_log, model_bytes, cache_bytes, hidden_bytes, get_filename, meta_to_cpu)

from deepspeed.runtime.utils import see_memory_usage


def get_ds_model(model_name, dtype, cpu_offload, disk_offload, offload_dir,
                     dummy_weights):
    import deepspeed
    import torch.distributed as dist
    from transformers.deepspeed import HfDeepSpeedConfig

    config = AutoConfig.from_pretrained(model_name)
    hidden_size = config.hidden_size
    deepspeed.init_distributed("nccl")
    rank = dist.get_rank()
    pin_memory = bool(args.pin_memory)

    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": 0, #2 * hidden_size * hidden_size,
            "stage3_param_persistence_threshold": hidden_size,
            "stage3_max_live_parameters": 2 * hidden_size * hidden_size,
        },
        "steps_per_print": 2000,
        "train_batch_size": args.batch_size,
        "wall_clock_breakdown": False,
    }

    if cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="cpu", pin_memory=pin_memory)

    if disk_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="nvme",
            pin_memory=True,
            nvme_path=offload_dir,
            buffer_count=5,
            buffer_size=2 * GB,
        )
        ds_config["aio"] = {
          "block_size": 1048576,
          "queue_depth": 8,
          "thread_count": 1,
          "single_submit": False,
          "overlap_events": True,
        }

    dschf = HfDeepSpeedConfig(ds_config)  # this tells from_pretrained to instantiate directly on gpus

    # clear cache / free memory
    torch.cuda.empty_cache()
    gc.collect()

    model = OPTForCausalLM.from_pretrained(
        dummy_weights or model_name, torch_dtype=dtype)
    model = model.eval()

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module
    print(f'model.config = {model.config}')

    return model


def get_hf_model(model_name, dtype, cpu_offload, disk_offload, offload_dir,
                     num_gpus, dummy_weights):
    if num_gpus == 1 and dtype != torch.int8:
        # Here we use a custom device_map instead of device_map == "auto"
        # becase we want to offload as many as possible weights out of GPU
        # to allow a larger batch size.
        if cpu_offload:
            # NOTE: We must put some weights on GPU. Otherwise, huggingface reports errors.
            device_map = {
                "model.decoder.embed_tokens.weight": 0,
                "model.decoder.embed_positions.weight": 0,
                "model.decoder.final_layer_norm": "cpu",
                "model.decoder.layers": "cpu",
                "lm_head.weight": 0,
            }
        elif disk_offload:
            device_map = {
                "model.decoder.embed_tokens.weight": 0,
                "model.decoder.embed_positions.weight": 0,
                "model.decoder.final_layer_norm": "disk",
                "model.decoder.layers": "disk",
                "lm_head.weight": 0,
            }
        else:
            device_map = None
        max_memory = None
    else:
        # Here we use device_map == "auto", but set a low `max_memory` threshold
        # becase we want to offload as many as possible weights out of GPU
        # to allow a larger batch size.
        device_map = "auto"
        if cpu_offload:
            # `max_memory` should be larger than the embedding.
            # We use 2GB here because the embeding of opt-175b is 1.2GB.
            max_memory = {k: "2GB" for k in range(num_gpus)}
        elif disk_offload:
            max_memory = {k: "2GB" for k in range(num_gpus)}
        else:
            max_memory = {k: "14GB" for k in range(num_gpus)}
        max_memory["cpu"] = "160GB"

    if dtype == torch.int8:
        kwargs = {"load_in_8bit": True}
    else:
        kwargs = {"torch_dtype": dtype}

    disable_torch_init()
    model = OPTForCausalLM.from_pretrained(dummy_weights or model_name,
        device_map=device_map, max_memory=max_memory,
        offload_folder=offload_dir, **kwargs)
    if device_map is None:
        model.cuda()

    model.eval()
    return model


def run_generation(model_name, batch_size, prompt_len, gen_len, cut_gen_len,
                   cpu_offload, disk_offload, offload_dir, use_int8,
                   num_nodes, num_gpus_per_node, use_deepspeed, dummy,
                   output_file, pkl_file, no_log, verbose, kv_offload):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name.replace("175b", "66b"), padding_side="left")

    # Load model
    if use_int8:
        dtype = torch.int8
    else:
        dtype = torch.float16

    config = AutoConfig.from_pretrained(model_name)

    if dummy:
        filename = os.path.join(offload_dir,
            f"{model_name.replace('/', '-')}-hf-weights/")
        if not os.path.exists(filename):
            print("create dummy weights")
            with init_empty_weights():
                model = OPTForCausalLM(config)
            model.save_pretrained(filename,
                state_dict=meta_to_cpu(model.state_dict(), torch.float16))
        dummy_weights = filename
    else:
        dummy_weights = None

    print("load model")
    if use_deepspeed:
        model = get_ds_model(model_name, dtype, cpu_offload, disk_offload,
            offload_dir, dummy_weights)
    else:
        model = get_hf_model(model_name, dtype, cpu_offload, disk_offload,
            offload_dir, num_gpus_per_node, dummy_weights)

    # Run generation
    execute_gen_len = cut_gen_len if cut_gen_len else gen_len
    if use_deepspeed:
        prompts = ["Paris is the capital city of"] * (batch_size // WORLD_SIZE)
    else:
        prompts = ["Paris is the capital city of"] * batch_size
    input_ids = tokenizer(prompts, return_tensors="pt",
                          padding="max_length",
                          max_length=prompt_len).input_ids.cuda()

    # set kv_offload in model config
    if kv_offload:
        model.config.kv_offload = True

    print(model, model.config)


    # add timing hooks
    def add_model_hooks(model):
        def start_time_hook(module, input):
            if hasattr(module, 'stage') and module.stage == "decode":
                return
            elif hasattr(module, 'stage') and module.stage == 'prefill':
                torch.cuda.synchronize()
                module.__start_time__ = time.time()

        def end_time_hook(module, input, output):
            if hasattr(module, 'stage') and module.stage == "decode":
                return
            elif hasattr(module, 'stage') and module.stage == 'prefill':
                torch.cuda.synchronize()
                module.__duration__ = time.time() - module.__start_time__
                module.stage = "decode"

        if not hasattr(model, '__start_time_hook_handle'):
            model.__start_time_hook_handle__ = model.register_forward_pre_hook(
                start_time_hook,
            )

        if not hasattr(model, '__end_time_hook_handle__'):
            model.__end_time_hook_handle__ = model.register_forward_hook(
                end_time_hook,
            )
    add_model_hooks(model)

    def set_model_stage(model, stage):
        model.stage = stage

    # Run
    print(f"benchmark, {execute_gen_len}, {input_ids.shape}")
    generate_kwargs = dict(max_new_tokens=execute_gen_len, do_sample=False)
    prefill_timings = []
    timer = timers("generate-forward")
    for _ in range(3):
        timer.start(sync_func=torch.cuda.synchronize)
        with torch.no_grad():
            set_model_stage(model, "prefill")
            output_ids = model.generate(input_ids=input_ids, **generate_kwargs)
            prefill_timings.append(model.__duration__)
        timer.stop(sync_func=torch.cuda.synchronize)
    costs = timers("generate-forward").costs

    if use_deepspeed and args.local_rank != 0:
        return

    def remove_model_hooks(module):
        if hasattr(module, '__start_time_hook_handle__'):
            module.__start_time_hook_handle__.remove()
            del module.__start_time_hook_handle__
        if hasattr(module, '__end_time_hook_handle__'):
            module.__end_time_hook_handle__.remove()
            del module.__end_time_hook_handle__
        if hasattr(module, 'stage'):
            del module.stage
        if hasattr(module, '__duration__'):
            del module.__duration__

    # Log output
    print("log")
    print(f'costs = {costs}, prefill_timings = {prefill_timings}')
    total_latency = costs[-1]
    prefill_latency = prefill_timings[-1] # np.mean(prefill_timings)
    remove_model_hooks(model)
    prefill_throughput = batch_size * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = total_latency - prefill_latency
    decode_throughput = batch_size * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = batch_size * gen_len
    total_throughput = num_generated_tokens / total_latency
    gpu_peak_mem = torch.cuda.max_memory_allocated(torch.device("cuda"))
    out_str = ""

    if verbose >= 2:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, (len(outputs)-1)//2, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += 70 * '-' + "\n"
        print(show_str)

        # Check lengths
        input_lens = [len(x) for x in input_ids]
        output_lens = [len(x) for x in output_ids]
        assert all(x == prompt_len for x in input_lens)
        assert all(x == prompt_len + execute_gen_len for x in output_lens)

    if args.log_file == "auto":
        filename = get_filename(model_name, batch_size, prompt_len,
            gen_len, cpu_offload, disk_offload, num_nodes,
            num_gpus_per_node, use_deepspeed) + ".log"
    else:
        filename = args.log_file

    projected = bool(cut_gen_len)
    cache_size = cache_bytes(config, batch_size, prompt_len + gen_len)
    hidden_size = hidden_bytes(config, batch_size, prompt_len + gen_len)
    log_str = write_benchmark_log(filename,
        model_bytes(config), cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if verbose >= 1:
        print(log_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--dummy", action="store_true",
        help="Use dummy weights for benchmark purposes.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--pin-memory", type=int, default=1)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--disk-offload", action="store_true")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir")
    parser.add_argument("--kv-offload", action="store_true", help="Use kv offload to cpu.")
    parser.add_argument("--int8", action="store_true")

    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--pkl-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    args = parser.parse_args()

    assert not (args.no_log and
                (args.output_file != "auto" or args.pkl_file != "auto"))

    if args.local_rank is None:  # huggingface
        use_deepspeed = False
        num_gpus_per_node = args.num_gpus
        num_nodes = 1
    else:  # deepspeed
        use_deepspeed = True
        WORLD_SIZE = int(os.getenv("WORLD_SIZE"))
        num_gpus_per_node = torch.cuda.device_count()
        num_nodes = WORLD_SIZE // num_gpus_per_node

    run_generation(args.model, args.batch_size, args.prompt_len, args.gen_len,
                   args.cut_gen_len, args.cpu_offload, args.disk_offload,
                   os.path.abspath(os.path.expanduser(args.offload_dir)),
                   args.int8, num_nodes, num_gpus_per_node, use_deepspeed,
                   args.dummy, args.log_file, args.pkl_file,
                   args.no_log, args.verbose, args.kv_offload)
