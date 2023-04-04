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

import numpy as np

from accelerate import (infer_auto_device_map, init_empty_weights,
    load_checkpoint_and_dispatch)
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from models.modeling_opt import OPTForCausalLM
import torch

from flexgen.timer import timers
from flexgen.utils import (GB, project_decode_latency,
    write_benchmark_log)
from flexgen.opt_config import (get_opt_config,
    disable_torch_init, disable_hf_opt_init)

from deepspeed.runtime.utils import see_memory_usage

def get_filename(model_name, batch_size, prompt_len, gen_len,
                 cpu_offload, disk_offload, num_nodes, num_gpus_per_node,
                 use_deepspeed):
    modelsize = model_name.split('-')[-1]
    if use_deepspeed:
        filename = "ds-"
    else:
        filename = "hf-"
    filename += f"{modelsize}-bs{batch_size}-prompt{prompt_len}-gen{gen_len}-"
    filename += f"n{num_nodes}x{num_gpus_per_node}-"
    if cpu_offload:
        filename += "cpu"
    elif disk_offload:
        filename += "disk"
    else:
        filename += "gpu"
    return filename


def meta_to_cpu(container, dtype=None):
    if isinstance(container, torch.Tensor):
        return torch.empty(*container.shape, dtype=dtype or container.dtype)
    elif isinstance(container, tuple):
        return tuple(meta_to_cpu(x, dtype) for x in container)
    elif isinstance(container, dict):
        return dict((k, meta_to_cpu(v, dtype)) for k, v in container.items())
    else:
        raise ValueError(f"Invalid type: {container}")


def realize_meta_module(module, dtype=None, device=None):
    for name, child in module.named_children():
        realize_meta_module(child, dtype, device)

    keys = list(module._parameters.keys())
    for k in keys:
        v = module._parameters[k]
        if v is not None:
            module._parameters[k] = torch.nn.Parameter(
                torch.empty(*v.shape, dtype=dtype or v.dtype,
                    device=device or v.device))

    keys = list(module._buffers.keys())
    for k in keys:
        v = module._buffers[k]
        assert v is None


def get_model_config(model_name):
    if "175b" in model_name:
        config = AutoConfig.from_pretrained("facebook/opt-66b")
        config.hidden_size = 12288
        config.word_embed_proj_dim = 12288
        config.ffn_dim = 12288 * 4
        config.num_attention_heads = 96
        config.num_hidden_layers = 96
    else:
        config = AutoConfig.from_pretrained(model_name)

    return config


def get_ds_opt_model(model_name, dtype, cpu_offload, disk_offload, offload_dir,
                     dummy_weights):
    import deepspeed
    import torch.distributed as dist
    from transformers.deepspeed import HfDeepSpeedConfig

    config = get_model_config(model_name)
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

    dschf = HfDeepSpeedConfig(ds_config)

    model = OPTForCausalLM.from_pretrained(
        dummy_weights or model_name, torch_dtype=dtype)
    model = model.eval()

    # TODO: this is a workaround for torch.empty not working with dtype=torch.float32 in init
    decoder = model.model.decoder
    pin_buffer_shape = [len(decoder.layers), 2, decoder.max_batch_size, decoder.num_heads, decoder.max_prompt_len + decoder.max_new_tokens - 1, decoder.hidden_dim_per_head]
    decoder.past_key_values_pin_mem = torch.empty(pin_buffer_shape, dtype=torch.float32, device='cpu', pin_memory=True)

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module
    print(f'model.config = {model.config}')

    return model


def get_hf_opt_model(model_name, dtype, cpu_offload, disk_offload, offload_dir,
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

    if dummy:
        config = get_model_config(model_name)
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
        model = get_ds_opt_model(model_name, dtype, cpu_offload, disk_offload,
            offload_dir, dummy_weights)
    else:
        model = get_hf_opt_model(model_name, dtype, cpu_offload, disk_offload,
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

    # Warmup
    # print("wamup")
    # generate_kwargs_warmup = dict(max_new_tokens=1, do_sample=False)
    # with torch.no_grad():
    #     output_ids = model.generate(input_ids=input_ids, **generate_kwargs_warmup)

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

    # decoder = model.model.decoder
    # pin_buffer_shape = [len(decoder.layers), 2, decoder.max_batch_size, decoder.num_heads, decoder.max_prompt_len + decoder.max_new_tokens - 1, decoder.hidden_dim_per_head]

    # Run
    print(f"benchmark, {execute_gen_len}, {input_ids.shape}")
    generate_kwargs = dict(max_new_tokens=execute_gen_len, do_sample=False)
    prefill_timings = []
    timer = timers("generate-forward")
    for _ in range(3):
        timer.start(sync_func=torch.cuda.synchronize)
        with torch.no_grad():
            set_model_stage(model, "prefill")
            # if decoder.past_key_values_pin_mem is not None:
                # del decoder.past_key_values_pin_mem
            # decoder.past_key_values_pin_mem = torch.empty(pin_buffer_shape, dtype=torch.float32, device='cpu', pin_memory=True)
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
    opt_config = get_opt_config(args.model)
    cache_size = opt_config.cache_bytes(batch_size, prompt_len + gen_len)
    hidden_size = opt_config.hidden_bytes(batch_size, prompt_len + gen_len)
    log_str = write_benchmark_log(filename,
        opt_config.model_bytes(), cache_size, hidden_size,
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
