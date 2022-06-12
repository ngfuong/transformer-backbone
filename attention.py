import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim*heads == embed_size), "Embedding size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embeddings into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys .reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape = (N, query_len, heads, head_dim)
        # keys shape = (N, key_len, heads, head_dim)
        # energy shape = (N, heads, query_len, key_len)
        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-inf"))

        attention = torch.softmax(energy*(self.embed_size**-0.5), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        # attention shape: (N, h, q, k)
        # value shape: (N, v, h, d) key_len and value_len matches now
        # out: (N, q, h, d) -> flatten to (N, q, h*d)

        out = self.fc_out(out)
        return out

# class TransformerBlock(nn.Module):



