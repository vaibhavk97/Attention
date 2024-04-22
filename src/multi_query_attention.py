from torch import nn
import torch
import math

class MultiQueryAttention(nn.Module):
    def __init__(self,dim:int, num_heads:int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim//num_heads
        assert dim%num_heads==0, "Fault in dimensions"
        self.wq = nn.Linear(dim,dim)
        self.wk = nn.Linear(dim,self.head_dim)
        self.wv = nn.Linear(dim,self.head_dim)

    def forward(self, q:torch.Tensor,k:torch.Tensor,v:torch.Tensor)->torch.Tensor:
        batch_size = q.shape[0]
        seq_len = q.shape[1]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = q.contiguous().view(batch_size,self.num_heads,seq_len, self.dim//self.num_heads)
        k = k.contiguous().view(batch_size,1,seq_len, self.dim//self.num_heads)
        v = v.contiguous().view(batch_size,1,seq_len, self.dim//self.num_heads)
        att = q@k.transpose(2,3)/math.sqrt(self.dim)
        att = torch.nn.functional.softmax(att,dim=-1)
        att = att@v
        att = att.transpose(1,2)
        att = att.reshape(batch_size,seq_len,-1)
        return att