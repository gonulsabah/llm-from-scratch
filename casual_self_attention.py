import torch
import torch.nn as nn


class CasualSelfAttention(nn.Module):
    def __init__(self, embedding_dim, output_dim, dropout_rate=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.q_weights = nn.Linear(embedding_dim, output_dim, bias=False)
        self.k_weights = nn.Linear(embedding_dim, output_dim, bias=False)
        self.v_weights = nn.Linear(embedding_dim, output_dim, bias=False)

        self.dropout = nn.Dropout(dropout_rate)
        # modelin bi parametresi degil gerekli olunca cagrilir
        self.register_buffer("mask", torch.tril(
            torch.ones(output_dim, output_dim)))

    def forward(self, x):
        q = self.q_weights(x)
        k = self.k_weights(x)
        v = self.v_weights(x)

        attention_scores = q@k.T
        # masked
        masked_attention_scores = attention_scores.masked_fill(
            self.mask == 0, -torch.inf)
        # softmax
        softed_attention_scores = torch.softmax(
            masked_attention_scores/(k.shape[-1] ** 0.5), dim=1)
        # droput
        dropouted_attention_scores = self.dropout(softed_attention_scores)

        return dropouted_attention_scores @ v
