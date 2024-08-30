import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, d_embed=512):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed

    def forward(self, x):
        out = self.embedding(x) * (self.d_embed ** 0.5)
        return out
