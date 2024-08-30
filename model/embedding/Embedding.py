import torch
import torch.nn as nn

from model.embedding.PositionalEncoding import PositionalEncoding
from model.embedding.TokenEmbedding import TokenEmbedding


class Embedding(nn.Module):

    def __init__(self, vocab_size, d_embed=512, max_len=256):
        super(Embedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size=vocab_size, d_embed=d_embed)
        self.pos_emb = PositionalEncoding(d_embed=d_embed,max_len=max_len)

    def forward(self, x):
        out = self.tok_emb(x)
        out = self.pos_emb(out)
        return out
