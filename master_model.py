import torch
import torch.nn as nn
from self_attention import SelfAttention
from casual_self_attention import CasualSelfAttention
from multi_head_attention import MultiHeadAttention
from norm_layer import LayerNorm
from mlp import MLP
from decoder_block import DecoderBlock


def get_rotary_position_encoding(input: torch.Tensor, base=10000, device="cpu"):
    context_length, dimension = input.shape  # 32, 4
    assert dimension % 2 == 0
    half_dimension = dimension // 2
    freqs_indices = torch.arange(
        0, half_dimension, device=device, dtype=torch.float32)

    freqs = 1.0/(base ** (freqs_indices/dimension))

    positions = torch.arange(
        0, context_length, device=device, dtype=torch.float32).unsqueeze(1)

    angles = positions * freqs

    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)

    input_even = input[:, 0:dimension:2]  # [0,2,4,..]
    input_odd = input[:, 1:dimension:2]  # [1,3,5,7,..]

    input_even_rotated = input_even * cos_angles - input_odd * sin_angles
    input_odd_rotated = input_even * sin_angles + input_odd * cos_angles

    input_rotated = torch.empty_like(input)
    input_rotated[:, 0:dimension:2] = input_even_rotated
    input_rotated[:, 1:dimension:2] = input_odd_rotated

    return input_rotated


class MasterModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, context_length, num_layers, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(context_length, embedding_dim)
        self.get_pos = get_rotary_position_encoding
        # 1 self.self_attention = SelfAttention(embedding_dim, embedding_dim)
        # 2self.self_attention = CasualSelfAttention(
        #    embedding_dim, embedding_dim, context_length, dropout_rate=0.5)
        # self.self_attention = MultiHeadAttention(
        #    embedding_dim,
        #    embedding_dim,
        #    context_length,
        #    num_heads,
        #    dropout_rate=0.5
        # )
        # self.norm = LayerNorm(embedding_dim)
        #     self.mlp= MLP(embedding_dim, embedding_dim)
        self.layers = nn.Sequential(*[
            DecoderBlock(embedding_dim, num_heads,
                         context_length, dropout_rate)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # dictionary meaning of the tokens(words)
        # meaning of the tokens in the sentece according to their position
        x = self.get_pos(x)
        x = self.layers(x)
        x = self.lm_head(x)
        return x
