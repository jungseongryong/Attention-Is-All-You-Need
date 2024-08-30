from torch import nn

from model.blocks.EncoderLayer import EncoderLayer
from model.embedding.Embedding import Embedding

class Encoder(nn.Module):

    def __init__(self, enc_vocab_size, max_len=256, n_layers=6, dff=2048, d_model=512, d_embed=512, n_heads=8, dropout=0.1):
        super().__init__()

        self.emb = Embedding(vocab_size=enc_vocab_size)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(EncoderLayer())
                        
    def forward(self, x, src_mask):
        out = self.emb(x)

        for layer in self.layers:
            out = layer(out, src_mask)

        return out
