import torch
import torch.nn as nn
from self_attention import SelfAttention  # DecoderBlock eklendi
from casual_self_attention import CasualSelfAttention  # DecoderBlock eklendi
from multi_head_attention import MultiHeadAttention  # DecoderBlock eklendi
from norm_layer import LayerNorm  # DecoderBlock eklendi
from mlp import MLP
from decoder_block import DecoderBlock
from master_embedding import MasterEmbedding


class MasterModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, context_length, num_layers, dropout_rate=0.5):
        super().__init__()
        self.embedding = MasterEmbedding(vocab_size, embedding_dim)
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

        # x = self.get_pos(x) # meaning of the tokens in the sentece according to their position
        x = self.layers(x)
        x = self.lm_head(x)
        return x
