from torch import nn
import math
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self,dim:int, num_heads:int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim%num_heads==0, "Fault in dimensions"
        self.wk = nn.Linear(dim,dim)
        self.wq = nn.Linear(dim,dim)
        self.wv = nn.Linear(dim,dim)

    def forward(self, q:torch.Tensor,k:torch.Tensor,v:torch.Tensor)->torch.Tensor:
        batch_size = q.shape[0]
        num_heads = self.num_heads
        dim = self.dim
        seq_len = q.shape[1]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = q.reshape(batch_size,num_heads,seq_len, dim//num_heads)
        k = k.reshape(batch_size,num_heads,seq_len, dim//num_heads)
        v = v.reshape(batch_size,num_heads,seq_len, dim//num_heads)
        att = q@k.transpose(2,3)/math.sqrt(dim)
        att = torch.nn.functional.softmax(att,dim=-1)
        att = att@v
        att = att.transpose(1,2)
        att = att.reshape(batch_size,seq_len,-1)
        return att