from torch import nn

from model.blocks.DecoderLayer import DecoderLayer
from model.embedding.Embedding import Embedding

class Decoder(nn.Module):

    def __init__(self, dec_vocab_size, max_len=256, n_layers=6, dff=2048, d_model=512, d_embed=512, n_heads=8, dropout=0.1):
        super().__init__()

        self.emb = Embedding(vocab_size=dec_vocab_size)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(DecoderLayer())
                        
        self.linear = nn.Linear(d_model, dec_vocab_size)
      
    def forward(self, x, c, trg_mask, src_mask):

        out = self.emb(x)

        for layer in self.layers:
            out = layer(out, c, trg_mask, src_mask)

        out = self.linear(out)
        return out
