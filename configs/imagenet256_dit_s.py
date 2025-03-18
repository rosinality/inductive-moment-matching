from functools import partial

from slickconf import field
from torch import optim

from imm import loss, model
from imm.data import CenterCrop
from imm.dit import DiT
from imm.vae import load_diffusers_vae, VAE

conf = field()

eta_min = 0
eta_max = 160
eps_k = 2**12
alpha_fn = partial(loss.ot_fm_alpha)
sigma_fn = partial(loss.ot_fm_sigma)
sigma_data = 0.5

conf.model = field(
    latent=VAE(
        load_diffusers_vae("stabilityai/sd-vae-ft-ema"),
        latent_mean=(0.86488, -0.27787343, 0.21616915, 0.3738409),
        latent_std=(4.85503674, 5.31922414, 3.93725398, 3.9870003),
        out_mean=0,
        out_std=0.5,
    ),
    model=DiT(
        image_size=256 // 8,
        patch_size=2,
        in_dim=4,
        dim=384,
        n_layers=12,
        n_heads=6,
        n_classes=1000,
    ),
    preconditioner=partial(
        model.IMMPreconditioner,
        c_in_fn=partial(model.common_c_in, sigma_data=sigma_data),
        c_out_fn=partial(model.euler_fm_c_out, sigma_data=sigma_data),
        c_skip_fn=partial(model.euler_fm_c_skip, sigma_data=sigma_data),
        c_noise_fn=partial(model.common_c_noise, timescale=1000),
        alpha_fn=alpha_fn,
        sigma_fn=sigma_fn,
        time_embed_type="identity",
    ),
)


conf.training = field(
    image_transform=CenterCrop(256),
    optimizer=partial(
        optim.AdamW,
        lr=1e-4,
    ),
    weight_decay=0.0,
    criterion=loss.IMMLoss(
        alpha_fn=alpha_fn,
        sigma_fn=sigma_fn,
        kernel_fn=partial(
            loss.laplace_kernel,
            weight_fn=partial(model.euler_fm_c_out_inv, sigma_data=sigma_data),
        ),
        mapping_fn=partial(
            loss.mapping_constant_decrement,
            eta=partial(loss.ot_fm_eta),
            eta_inv=partial(loss.ot_fm_eta_inv),
            eps=(eta_max - eta_min) / eps_k,
            delta=1e-4,
        ),
        log_snr_fn=partial(loss.ot_fm_log_snr),
        d_log_snr_fn=partial(loss.ot_fm_d_log_snr),
        weight_a=2,
        weight_b=4,
        time_min=0,
        time_max=0.994,
    ),
    n_iters=1200000,
    train_batch_size=4096,
    label_dropout=0.1,
    group_size=4,
    sigma_data=sigma_data,
    ema=0.9999,
)

conf.data = field()

conf.output = field(
    save_step=1000,
    output_dir="imagenet256_dit_s",
    wandb_project="inductive-moment-matching",
    wandb_name="imagenet256_dit-s",
)
