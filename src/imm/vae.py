import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, vae, latent_mean, latent_std, out_mean, out_std):
        super().__init__()

        self.vae = vae
        self.scale = torch.tensor(out_std, dtype=torch.float32) / torch.tensor(
            latent_std, dtype=torch.float32
        )
        self.bias = (
            torch.tensor(out_mean, dtype=torch.float32)
            - torch.tensor(latent_mean, dtype=torch.float32) * self.scale
        )

    @property
    def dtype(self):
        dtype = None

        for p in self.vae.parameters():
            dtype = p.dtype

            if p.is_floating_point():
                return dtype

        return dtype

    def encode_images(self, input):
        input = input.to(torch.float32) / 127.5 - 1
        out = self.vae.encode(input.to(self.dtype))["latent_dist"]

        return out.mean, out.std

    def encode_latents(self, mean, std):
        dtype = mean.dtype
        mean = mean.to(torch.float32)
        std = std.to(torch.float32)
        out = mean + torch.randn_like(std) * std
        out = out * self.scale.to(out.device).reshape(1, -1, 1, 1)
        out = out + self.bias.to(out.device).reshape(1, -1, 1, 1)

        return out.to(dtype)


def load_diffusers_vae(name, device="cuda", dtype=torch.float16):
    import diffusers

    vae = diffusers.models.AutoencoderKL.from_pretrained(name, torch_dtype=dtype)

    vae = torch.compile(
        vae.eval().requires_grad_(False).to(device), mode="max-autotune", fullgraph=True
    )

    return vae
