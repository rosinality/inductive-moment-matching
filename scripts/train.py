from copy import deepcopy
import os

from slickconf import load_arg_config, instantiate, summarize
import torch
from torch import distributed as dist
from torch.distributed.tensor import DTensor
from torch import amp
from torch.utils import data
from tqdm import tqdm

try:
    import wandb

except ImportError:
    wandb = None

from imm.config import IMMConfig
from imm.data import LatentDataset
from imm.fsdp import apply_compile, apply_ddp, apply_fsdp
from imm.logger import get_logger
from imm.loss import ddim_interpolant
from imm.parallel_dims import ParallelDims


def infinite_loader(loader):
    iterator = iter(loader)

    while True:
        try:
            yield next(iterator)

        except StopIteration:
            iterator = iter(loader)


@torch.no_grad()
def update_ema(ema_model, model, decay):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())

    for name, param in model_params.items():
        if not ema_params[name].requires_grad:
            continue

        ema_params[name].detach().mul_(decay).add_(param.detach(), alpha=1 - decay)


def get_full_state_dict(state_dict):
    full_state_dict = {}

    for k, v in state_dict.items():
        if isinstance(v, DTensor):
            v = v.full_tensor()

        full_state_dict[k] = v

    return full_state_dict


def main():
    conf = load_arg_config(IMMConfig)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0 and conf.output.wandb_project is not None:
        wandb.init(project=conf.output.wandb_project, name=conf.output.wandb_name)

    pdims = ParallelDims(
        dp_replicate=world_size,
        dp_shard=1,
        tp=1,
        pp=1,
    )
    mesh = pdims.build_mesh("cuda")
    logger = get_logger(mesh)

    logger.info(summarize(conf))
    device = torch.device("cuda")

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    torch.cuda.set_device(torch.cuda.device(rank))

    latent_encoder = instantiate(conf.model.latent, None)
    model = instantiate(conf.model.model)
    apply_compile(model)
    apply_fsdp(model, mesh, param_dtype="float16", reduce_dtype="float32")
    model = instantiate(conf.model.preconditioner)(model)
    # apply_ddp(model, mesh, compile=True)
    model = model.to(device)

    if conf.training.ema is not None:
        model_ema = instantiate(conf.model.model)
        apply_compile(model_ema)
        apply_fsdp(model_ema, mesh, param_dtype="float16", reduce_dtype="float32")

    optimizer = instantiate(conf.training.optimizer)(model.parameters())
    criterion = instantiate(conf.training.criterion)
    scaler = amp.GradScaler()

    dataset = LatentDataset(conf.data.latent_dataset)
    sampler = data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )
    loader = data.DataLoader(
        dataset,
        batch_size=conf.training.train_batch_size // world_size,
        sampler=sampler,
        num_workers=8,
        multiprocessing_context="fork",
    )

    loader = infinite_loader(loader)

    running_loss = 0
    for i in range(conf.training.n_iters):
        latent_mean, latent_std, label, _ = next(loader)
        latent = latent_encoder.encode_latents(
            latent_mean.to(device), latent_std.to(device)
        ).to(torch.float32)
        label = label.to(device)

        is_uncond = None
        if conf.training.label_dropout > 0:
            is_uncond = (
                torch.rand(label.shape[0], device=device) < conf.training.label_dropout
            )

        time_s, time_r, time_t = criterion.sample_timesteps(
            latent.shape[0] // conf.training.group_size, device=device
        )

        time_s_repeated = time_s.repeat_interleave(conf.training.group_size)
        time_r_repeated = time_r.repeat_interleave(conf.training.group_size)
        time_t_repeated = time_t.repeat_interleave(conf.training.group_size)

        latent_t = ddim_interpolant(
            torch.randn_like(latent) * conf.training.sigma_data,
            latent,
            time_t_repeated,
            time_t_repeated.new_tensor((1,)),
            criterion.alpha_fn,
            criterion.sigma_fn,
        )
        latent_r = ddim_interpolant(
            latent_t,
            latent,
            time_r_repeated,
            time_t_repeated,
            criterion.alpha_fn,
            criterion.sigma_fn,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            out_stop = model(
                latent_r,
                time_r_repeated,
                time_s_repeated,
                condition_embed=label,
                is_uncond=is_uncond,
            )
            out_stop = out_stop.reshape(
                -1, conf.training.group_size, *out_stop.shape[1:]
            )

        out_running = model(
            latent_t,
            time_t_repeated,
            time_s_repeated,
            condition_embed=label,
            is_uncond=is_uncond,
        )
        out_running = out_running.reshape(
            -1, conf.training.group_size, *out_running.shape[1:]
        )

        loss = criterion(out_running, out_stop, time_s, time_r, time_t)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if conf.training.ema is not None:
            update_ema(model_ema, model.model, conf.training.ema)

        if (i + 1) % conf.output.log_step == 0:
            avg_loss = torch.tensor(running_loss / conf.output.log_step, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / world_size
            running_loss = 0

            logger.info(f"[{i + 1}/{conf.training.n_iters}] loss: {avg_loss}")

            if rank == 0 and conf.output.wandb_project is not None:
                wandb.log({"train/loss": avg_loss}, step=i + 1)

        if conf.output.save_step is not None and (i + 1) % conf.output.save_step == 0:
            checkpoint = {
                "model": get_full_state_dict(model.state_dict()),
                "model_ema": get_full_state_dict(model_ema.state_dict()),
                "optimizer": get_full_state_dict(optimizer.state_dict()),
                "scaler": scaler.state_dict(),
                "conf": conf.dict(),
            }

            if rank == 0:
                os.makedirs(conf.output.output_dir, exist_ok=True)

                torch.save(
                    checkpoint,
                    os.path.join(conf.output.output_dir, f"model_{i + 1:07d}.pt"),
                )


if __name__ == "__main__":
    main()

    torch.distributed.destroy_process_group()
