import torch
from torch import nn
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_groups: int, device: torch.device):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_kv_heads = num_heads // num_groups
        self.head_dim = dim // num_heads
        self.wq = nn.Parameter(torch.randn(dim, dim, device=device))
        self.wk = nn.Parameter(
            torch.randn(self.num_kv_heads, dim, self.head_dim, device=device)[None, :, :, :]
            .expand(self.num_groups, -1, -1, -1)
            .reshape(-1, dim, self.head_dim)
            .transpose(2, 1)
            .reshape(dim, dim) # This does not seem to be parameterically efficient as reshape creates copies.
        )
        self.wv = nn.Parameter(
            torch.randn(self.num_kv_heads, dim, self.head_dim, device=device)[None, :, :, :]
            .expand(self.num_groups, -1, -1, -1)
            .reshape(-1, dim, self.head_dim)
            .transpose(2, 1)
            .reshape(dim, dim)
        ) # This does not seem to be parameterically efficient as reshape creates copies.
        self.to(device)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size = q.shape[0]
        seq_len = q.shape[1]
        q, k, v = q.to(self.wq.device), k.to(self.wk.device), v.to(self.wv.device)
        k = k @ self.wk
        q = q @ self.wq
        v = v @ self.wv
        q = q.reshape(batch_size, self.num_heads, seq_len, self.dim // self.num_heads)
        k = k.reshape(batch_size, self.num_heads, seq_len, self.dim // self.num_heads)
        v = v.reshape(batch_size, self.num_heads, seq_len, self.dim // self.num_heads)
        att = q @ k.transpose(2, 3) / math.sqrt(self.dim)
        att = torch.nn.functional.softmax(att, dim=-1)
        att = att @ v
        att = att.transpose(1, 2)
        att = att.reshape(batch_size, seq_len, -1)
        return att