import torch
from torch import nn


def common_c_in(s, t, alpha_s, alpha_t, sigma_s, sigma_t, sigma_data):
    return torch.rsqrt(alpha_t**2 + sigma_t**2) / sigma_data


def euler_fm_c_out(s, t, alpha_s, alpha_t, sigma_s, sigma_t, sigma_data):
    return -(t - s) * sigma_data


def euler_fm_c_out_inv(s, t, sigma_data):
    return 1 / ((t - s) * sigma_data)


def euler_fm_c_skip(s, t, alpha_s, alpha_t, sigma_s, sigma_t, sigma_data):
    return 1


def common_c_noise(t, timescale):
    return t * timescale


class IMMPreconditioner(nn.Module):
    def __init__(
        self,
        model,
        c_in_fn,
        c_out_fn,
        c_skip_fn,
        c_noise_fn,
        alpha_fn,
        sigma_fn,
        time_embed_type="identity",
    ):
        super().__init__()

        self.model = model
        self.dtype = model.dtype

        self.c_in_fn = c_in_fn
        self.c_out_fn = c_out_fn
        self.c_skip_fn = c_skip_fn
        self.c_noise_fn = c_noise_fn
        self.alpha_fn = alpha_fn
        self.sigma_fn = sigma_fn

        self.time_embed_type = time_embed_type

    def forward(self, input, t, s, **rest_inputs):
        input = input.to(torch.float32)
        s = s.to(torch.float32).reshape(-1, 1, 1, 1)
        t = t.to(torch.float32).reshape(-1, 1, 1, 1)

        alpha_s = self.alpha_fn(s)
        alpha_t = self.alpha_fn(t)
        sigma_s = self.sigma_fn(s)
        sigma_t = self.sigma_fn(t)

        c_in = self.c_in_fn(s, t, alpha_s, alpha_t, sigma_s, sigma_t)
        c_out = self.c_out_fn(s, t, alpha_s, alpha_t, sigma_s, sigma_t)
        c_skip = self.c_skip_fn(s, t, alpha_s, alpha_t, sigma_s, sigma_t)

        if self.time_embed_type == "identity":
            c_noise_s = self.c_noise_fn(s)

        elif self.time_embed_type == "stride":
            c_noise_s = self.c_noise_fn(t - s)

        c_noise_t = self.c_noise_fn(t)

        model_out = self.model(
            (c_in * input).to(self.dtype),
            c_noise_t.to(self.dtype).flatten(),
            c_noise_s.to(self.dtype).flatten(),
            **rest_inputs,
        )
        out = c_skip * input + c_out * model_out.to(torch.float32)

        return out
