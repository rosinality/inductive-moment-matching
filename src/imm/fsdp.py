import torch
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._composable.replicate import replicate


def get_torch_dtype(dtype):
    if dtype is None:
        return None

    if isinstance(dtype, torch.dtype):
        return dtype

    if dtype in ["float32", "torch.float32", "fp32"]:
        return torch.float32

    if dtype in ["float16", "torch.float16", "half", "fp16"]:
        return torch.float16

    if dtype in ["bfloat16", "bfloat", "torch.bfloat16", "bf16"]:
        return torch.bfloat16

    raise ValueError(f"passed dtype {dtype} is not an appropriate dtype")


def apply_fsdp(model, dp_mesh, param_dtype, reduce_dtype):
    if param_dtype is not None:
        param_dtype = get_torch_dtype(param_dtype)

    if reduce_dtype is not None:
        reduce_dtype = get_torch_dtype(reduce_dtype)

    mixed_precision = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype
    )

    for i, block in model.blocks.items():
        reshard_after_forward = int(i) < len(model.blocks) - 1

        fully_shard(
            block,
            mesh=dp_mesh,
            mp_policy=mixed_precision,
            reshard_after_forward=reshard_after_forward,
        )

    fully_shard(
        model, mesh=dp_mesh, mp_policy=mixed_precision, reshard_after_forward=True
    )

    return model


def apply_ddp(model, dp_mesh, compile=False, compile_autograd=False):
    if compile:
        if compile_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )

        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)


def apply_compile(model):
    for i, block in model.blocks.named_children():
        block = torch.compile(block, fullgraph=True)
        model.blocks.register_module(i, block)
