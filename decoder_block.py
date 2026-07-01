
import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from norm_layer import LayerNorm
from mlp import MLP


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length, dropout_rate=0.5):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            embedding_dim, embedding_dim, context_length, num_heads, dropout_rate=0.5)
        self.norm1 = LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

    def forward(self, x):
        res = x  # ilk inputi tutuyoruz, residual connection icin.
        res_norm = self.norm1(res)

        x = self.self_attention(x)
        x = self.norm1(x)

        x = x + res_norm  # residual connection ile birlestiriyoruz.

        res = self.norm2(x)
        x = self.mlp(x)
        x = self.norm2(x)

        x = x + res  # residual connection ile birlestiriyoruz.

        return x
