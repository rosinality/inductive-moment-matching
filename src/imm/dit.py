import math

import torch
from torch import nn
from torch.nn import functional as F


def affine_modulate(input, shift, scale):
    return input * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, dim, patch_size):
        super().__init__()

        self.patch_size = patch_size
        self.linear = nn.Linear(patch_size**2 * in_dim, dim, bias=True)

    def forward(self, input):
        batch, dim, height, width = input.shape
        out = (
            input.reshape(
                batch,
                dim,
                height // self.patch_size,
                self.patch_size,
                width // self.patch_size,
                self.patch_size,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(
                batch,
                -1,
                dim * (self.patch_size**2),
            )
        )
        out = self.linear(out)

        return out


class TimeEmbedding(nn.Module):
    def __init__(
        self, dim, freq_embed_dim=256, embed_type="positional", use_mlp=True, scale=1
    ):
        super().__init__()

        self.dim = dim

        self.mlp = None
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(freq_embed_dim, dim, bias=True),
                nn.SiLU(),
                nn.Linear(dim, dim, bias=True),
            )

        self.freq_embed_dim = freq_embed_dim
        self.embed_type = embed_type

        if self.embed_type == "fourier":
            self.register_buffer("freqs", torch.randn(freq_embed_dim // 2) * scale)

    def init_weights(self):
        if self.mlp is None:
            return

        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def positional_embed(self, t, max_period=10000):
        half = self.freq_embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float64, device=t.device)
            / half
        )
        args = t[:, None].to(torch.float64) * freqs[None]
        embed = torch.cat((torch.cos(args), torch.sin(args)), -1)

        if self.freq_embed_dim % 2 != 0:
            embed = torch.cat((embed, torch.zeros_like(embed[:, :1])), -1)

        return embed

    def fourier_embed(self, t):
        out = t.to(torch.float64).ger((2 * math.pi * self.freqs.to(torch.float64)))
        out = torch.cat((torch.cos(out), torch.sin(out)), 1)

        return out

    def forward(self, t):
        if self.embed_type == "positional":
            out = self.positional_embed(t, self.dim)

        elif self.embed_type == "fourier":
            out = self.fourier_embed(t)

        else:
            raise ValueError(f"Unknown embed type: {self.embed_type}")

        out = out.to(t.dtype)

        if self.mlp is not None:
            out = self.mlp(out)

        return out


class ClassEmbedding(nn.Module):
    def __init__(self, n_classes, dim):
        super().__init__()

        self.n_classes = n_classes
        self.embed = nn.Embedding(n_classes + 1, dim)

    def forward(self, condition: torch.Tensor, is_uncond: torch.Tensor | None = None):
        if is_uncond is not None:
            uncond_class = condition.new_full((condition.shape[0],), self.n_classes)
            condition = torch.where(is_uncond, uncond_class, condition)

        out = self.embed(condition)

        return out


class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads, qkv_bias=True, out_bias=True):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.out = nn.Linear(dim, dim, bias=out_bias)

    def forward(self, input):
        batch, length, dim = input.shape
        qkv = (
            self.qkv(input)
            .reshape(batch, length, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).reshape(batch, length, dim)
        out = self.out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, activation, dropout=0):
        super().__init__()

        self.linear1 = nn.Linear(dim, ff_dim, bias=True)
        self.activation = activation
        self.linear2 = nn.Linear(ff_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        out = self.linear1(input)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return out


class DiTBlock(nn.Module):
    def __init__(self, dim, n_heads, time_embed_dim, ff_ratio=4, skip=False, dropout=0):
        super().__init__()

        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(
            dim,
            n_heads,
            qkv_bias=True,
        )
        self.norm_ff = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        ff_dim = int(dim * ff_ratio)
        self.ff = FeedForward(dim, ff_dim, nn.GELU(approximate="tanh"), dropout=dropout)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(time_embed_dim, 6 * dim, bias=True)
        )
        self.skip = nn.Linear(2 * dim, dim) if skip else None

    def init_weights(self):
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, input, condition):
        shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = (
            self.modulation(condition).chunk(6, dim=1)
        )

        out = input + gate_attn.unsqueeze(1) * self.attn(
            affine_modulate(self.norm_attn(input), shift_attn, scale_attn)
        )
        out = out + gate_ff.unsqueeze(1) * self.ff(
            affine_modulate(self.norm_ff(out), shift_ff, scale_ff)
        )

        return out


class OutputBlock(nn.Module):
    def __init__(self, dim, out_dim, patch_size):
        super().__init__()

        self.patch_size = patch_size
        self.out_dim = out_dim

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size**2 * out_dim, bias=True)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2, bias=True))

    def init_weights(self):
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)
        nn.init.zeros_(self.linear.weight)

        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, input, condition):
        batch = input.shape[0]
        length = input.shape[1]
        height = width = int(length**0.5)

        shift, scale = self.modulation(condition).chunk(2, dim=1)
        out = affine_modulate(self.norm(input), shift, scale)
        out = self.linear(out)

        out = out.reshape(
            batch, height, width, self.patch_size, self.patch_size, self.out_dim
        )
        out = out.permute(0, 5, 1, 3, 2, 4)
        out = out.reshape(
            batch, self.out_dim, height * self.patch_size, width * self.patch_size
        )

        return out


class DiT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        in_dim,
        dim,
        n_layers,
        n_heads,
        ff_ratio=4,
        n_classes=None,
        s_embed=True,
        time_embed_multiplier=1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(in_dim, dim, patch_size)

        time_embed_dim = dim * time_embed_multiplier

        self.s_embed = None
        if s_embed:
            self.s_embed = TimeEmbedding(time_embed_dim)

        self.t_embed = TimeEmbedding(time_embed_dim)

        if n_classes is not None:
            self.condition_embed = ClassEmbedding(n_classes + 1, dim)

        self.register_buffer(
            "pos_embed",
            get_sinusoid_2d_pos_emb(dim, image_size // patch_size, extra_tokens=0)
            .to(torch.float32)
            .unsqueeze(0),
        )

        self.blocks = ModuleDict()
        for i in range(n_layers):
            self.blocks[str(i)] = DiTBlock(
                dim, n_heads, time_embed_dim, ff_ratio, skip=False
            )

        self.out_block = OutputBlock(dim, in_dim, patch_size)

        self.init_weights()

    @property
    def dtype(self):
        dtype = None

        for p in self.parameters():
            dtype = p.dtype

            if p.is_floating_point():
                return dtype

        return dtype

    def init_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init)

        if self.s_embed is not None:
            self.s_embed.init_weights()

        self.t_embed.init_weights()

        for block in self.blocks.values():
            block.init_weights()

        self.out_block.init_weights()

    def forward(self, input, time_t, time_s, condition_embed=None, is_uncond=None):
        out = self.patch_embed(input) + self.pos_embed

        t_embed = self.t_embed(time_t)

        if self.s_embed is not None:
            s_embed = self.s_embed(time_s)
            t_embed = t_embed + s_embed

        condition_embed = (
            self.condition_embed(condition_embed, is_uncond)
            if condition_embed is not None
            else 0
        )

        condition = condition_embed + t_embed

        for block in self.blocks.values():
            out = block(out, condition)

        out = self.out_block(out, condition)

        return out


def get_sinusoid_pos_1d_from_grid(dim, pos):
    omega = torch.arange(dim // 2, dtype=torch.float64)
    omega /= dim / 2.0
    omega = 1.0 / (10000**omega)

    pos = pos.reshape(-1)
    out = torch.outer(pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    return torch.cat((emb_sin, emb_cos), 1)


def get_sinusoid_pos_2d_from_grid(dim, grid):
    emb_h = get_sinusoid_pos_1d_from_grid(dim // 2, grid[0])
    emb_w = get_sinusoid_pos_1d_from_grid(dim // 2, grid[1])

    return torch.cat((emb_h, emb_w), 1)


def get_sinusoid_2d_pos_emb(dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = torch.arange(grid_size, dtype=torch.float64)
    grid_w = torch.arange(grid_size, dtype=torch.float64)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")
    grid = torch.stack(grid, 0)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    pos_embed = get_sinusoid_pos_2d_from_grid(dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = torch.cat((torch.zeros(extra_tokens, dim), pos_embed), 0)

    return pos_embed


# Copied from torch.nn.modules.module, required for a custom __repr__ for ModuleList
def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class ModuleDict(nn.ModuleDict):
    # taken from nn.ModuleList
    def __repr__(self):
        """Return a custom repr for ModuleDict that compresses repeated module representations."""
        list_of_reprs = [repr(item) for item in self.values()]
        if len(list_of_reprs) == 0:
            return self._get_name() + "()"

        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1

                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)

        lines = []
        main_str = self._get_name() + "("
        for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
            local_repr = f"({start_id}): {b}"  # default repr

            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {b}"

            local_repr = _addindent(local_repr, 2)
            lines.append(local_repr)

        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str
