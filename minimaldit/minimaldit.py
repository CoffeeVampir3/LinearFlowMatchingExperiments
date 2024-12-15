import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, dim: int, n_positions: int = 1024):
        super().__init__()
        pe = torch.zeros(n_positions, dim)
        position = torch.arange(0, n_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class PatchEmbed(nn.Module):
    """Convert image into patches and embed them"""
    def __init__(self, img_size: int = 256, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        # Project and reshape: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size) -> (B, embed_dim, n_patches)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x

class MinimalDiT(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8
    ):
        super().__init__()
        
        # Calculate number of patches
        n_patches = (img_size // patch_size) ** 2
        
        # Patch embedding layer
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(dim, n_patches)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(depth)
        ])
        
        # Final layer norm and projection
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, patch_size * patch_size * in_channels)
        
        # Save hyperparameters
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch embeddings back to image"""
        B, N, D = x.shape
        h = w = int(N ** 0.5)
        x = x.reshape(B, h, w, self.patch_size, self.patch_size, self.in_channels)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(B, self.in_channels, self.img_size, self.img_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert image to patches and embed: (B, C, H, W) -> (B, n_patches, dim)
        x = self.patch_embed(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final normalization and projection
        x = self.norm(x)
        x = self.to_out(x)
        
        # Convert patches back to image
        x = self.unpatchify(x)
        return x