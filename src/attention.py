import torch
from torch import nn
import math

class Attention(nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        self.dim = dim
        self.wk = nn.Linear(dim,dim)
        self.wq = nn.Linear(dim,dim)
        self.wv = nn.Linear(dim,dim)

    def forward(self, q:torch.Tensor,k:torch.Tensor,v:torch.Tensor)->torch.Tensor:
        k = self.wk(k)
        q = self.wq(q)
        v = self.wv(v)

        att = q@k.transpose(1,2)/math.sqrt(self.dim)
        att = torch.nn.functional.softmax(att,dim=-1)
        att = att@v
        return att