from typing import Callable

import torch
from torch import nn


def ddim_interpolant(x_t, x, s, t, alpha_fn, sigma_fn):
    alpha_t = alpha_fn(t).reshape(-1, 1, 1, 1)
    alpha_s = alpha_fn(s).reshape(-1, 1, 1, 1)
    sigma_t = sigma_fn(t).reshape(-1, 1, 1, 1)
    sigma_s = sigma_fn(s).reshape(-1, 1, 1, 1)

    return (alpha_s - sigma_s / sigma_t * alpha_t) * x + sigma_s / sigma_t * x_t


def laplace_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    time_s: torch.Tensor,
    time_t: torch.Tensor,
    weight_fn: Callable,
    eps: float = 1e-8,
) -> torch.Tensor:
    weight = weight_fn(time_s.to(torch.float64), time_t.to(torch.float64)).to(
        torch.float32
    )

    weight = weight.view(-1, *([1] * (x.ndim - 2)))
    n_elems = x.shape[-1]

    return torch.exp(-weight * torch.norm(x - y, p=2, dim=-1).clip(min=eps) / n_elems)


def ot_fm_alpha(t):
    return 1 - t


def ot_fm_sigma(t):
    return t


def ot_fm_eta(t):
    return t / (1 - t)


def ot_fm_eta_inv(eta):
    dtype = eta.dtype
    t = eta / (1 + eta)
    t = torch.nan_to_num(t, 1)
    t = t.to(dtype)

    return t


def ot_fm_log_snr(t):
    dtype = t.dtype
    t = t.to(torch.float64)

    log_snr = 2 * (torch.log(1 - t) - torch.log(t))

    log_snr = log_snr.to(dtype)

    return log_snr


def ot_fm_d_log_snr(t):
    dtype = t.dtype
    t = t.to(torch.float64)

    d_log_snr = -2 * (1 / (1 - t) + 1 / t)

    d_log_snr = d_log_snr.to(dtype)

    return d_log_snr


def mapping_constant_decrement(s, t, eta, eta_inv, eps, delta):
    r = torch.max(s, torch.min(t - delta, eta_inv(eta(t) - eps)))

    return r


class IMMLoss(nn.Module):
    def __init__(
        self,
        alpha_fn,
        sigma_fn,
        kernel_fn,
        mapping_fn,
        log_snr_fn,
        d_log_snr_fn,
        weight_a,
        weight_b,
        time_min,
        time_max,
    ):
        super().__init__()

        self.alpha_fn = alpha_fn
        self.sigma_fn = sigma_fn
        self.kernel_fn = kernel_fn
        self.mapping_fn = mapping_fn
        self.log_snr_fn = log_snr_fn
        self.d_log_snr_fn = d_log_snr_fn

        self.weight_a = weight_a
        self.weight_b = weight_b
        self.time_min = time_min
        self.time_max = time_max

    def sample_timesteps(
        self, batch_size: int, device: torch.device | str = "cuda"
    ) -> torch.Tensor:
        time_t = (
            torch.rand(batch_size, device=device) * (self.time_max - self.time_min)
            + self.time_min
        )
        time_s = torch.rand(batch_size, device=device)
        time_s = time_s * (time_t - self.time_min) + self.time_min
        time_r = self.mapping_fn(time_s, time_t)

        return time_s, time_r, time_t

    def get_weight(self, s, t):
        log_snr = self.log_snr_fn(t)
        d_log_snr = self.d_log_snr_fn(t)
        alpha_t = self.alpha_fn(t)
        sigma_t = self.sigma_fn(t)

        weight = (
            0.5
            * torch.sigmoid(self.weight_b - log_snr)
            * (-d_log_snr)
            * (alpha_t**self.weight_a)
            / (alpha_t**2 + sigma_t**2)
        )

        return weight

    def forward(self, f_running, f_stop, s, r, t):
        n_groups, group_size, *_ = f_running.shape

        weight = self.get_weight(s, t)

        term1 = self.kernel_fn(
            f_running.reshape(n_groups, group_size, 1, -1),
            f_running.reshape(n_groups, 1, group_size, -1),
            time_s=s,
            time_t=t,
        )

        term2 = self.kernel_fn(
            f_stop.reshape(n_groups, group_size, 1, -1),
            f_stop.reshape(n_groups, 1, group_size, -1),
            time_s=s,
            time_t=t,
        )

        term3 = self.kernel_fn(
            f_running.reshape(n_groups, group_size, 1, -1),
            f_stop.reshape(n_groups, 1, group_size, -1),
            time_s=s,
            time_t=t,
        )

        terms = term1 + term2 - 2 * term3
        out = terms.mean((1, 2))
        out = out * weight

        return out.mean()
