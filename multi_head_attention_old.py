import torch
import torch.nn as nn
from casual_self_attention import CasualSelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, output_dim, context_length, num_heads, dropout_rate=0.5):
        super().__init__()

        self.heads = nn.ModuleList(
            [CasualSelfAttention(
                embedding_dim, output_dim, context_length, dropout_rate) for _ in range(num_heads)]
        )
        self.projection = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        attention_outs = []
        for head in self.heads:
            head_out = head(x)
            attention_outs.append(head_out)
        concated_attention_outs = torch.concat(attention_outs, dim=1)
        return self.projection(concated_attention_outs)
