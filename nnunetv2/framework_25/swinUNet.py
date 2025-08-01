import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

# ---------------------------
# Swin Transformer Block
# ---------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_proj = nn.Linear(dim, dim)
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.attn_proj(out)

# dropout block can be added to the SwinBlock module
class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4.):
    # def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4., dropout_rate=0.1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(int(dim * mlp_ratio), dim)
            # nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x) + x
        x = self.norm2(x)
        x = self.mlp(x) + x
        return x


# ---------------------------
# Patch Embedding
# ---------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=1, embed_dim=96):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)  # B C H W
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x


# ---------------------------
# Patch Merging
# ---------------------------
class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # (B H/2 W/2 C)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4C
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


# ---------------------------
# Swin UNet Encoder-Decoder
# ---------------------------
class SwinUNet(nn.Module):
    def __init__(self, img_size=224, in_chans=1, num_classes=2, embed_dim=96):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim)
        self.stage1 = SwinBlock(embed_dim, img_size // 4, num_heads=3)
        self.down1 = PatchMerging(embed_dim)

        self.stage2 = SwinBlock(embed_dim * 2, img_size // 8, num_heads=6)
        self.down2 = PatchMerging(embed_dim * 2)

        self.bottleneck = SwinBlock(embed_dim * 4, img_size // 16, num_heads=12)

        self.up1 = nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2)
        self.decoder1 = nn.Conv2d(embed_dim * 4, embed_dim * 2, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=2, stride=2)
        self.decoder2 = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1)

        self.final = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]

        x = self.patch_embed(x)  # B, L, C
        x1 = self.stage1(x)
        x2 = self.down1(x1, H // 4, W // 4)

        x3 = self.stage2(x2)
        x4 = self.down2(x3, H // 8, W // 8)

        x5 = self.bottleneck(x4)

        # reshape
        x5_up = x5.transpose(1, 2).reshape(B, -1, H // 8, W // 8)
        x3_up = x3.transpose(1, 2).reshape(B, -1, H // 8, W // 8)
        x = torch.cat([self.up1(x5_up), x3_up], dim=1)
        x = self.decoder1(x)

        x1_up = x1.transpose(1, 2).reshape(B, -1, H // 4, W // 4)
        x = torch.cat([self.up2(x), x1_up], dim=1)
        x = self.decoder2(x)

        out = self.final(x)
        return out
