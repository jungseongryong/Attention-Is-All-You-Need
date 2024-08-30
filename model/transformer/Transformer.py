import torch
import torch.nn as nn


from model.transformer.Decoder import Decoder
from model.transformer.Encoder import Encoder

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_vocab_size, dec_vocab_size, vocab_size, n_layers=6, dff=2048, d_model=512, d_embed=512, n_heads=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.encoder = Encoder(enc_vocab_size, max_len=256, n_layers=6, dff=2048, d_model=512, d_embed=512, n_heads=8, dropout=0.1)
        self.decoder = Decoder(dec_vocab_size, max_len=256, n_layers=6, dff=2048, d_model=512, d_embed=512, n_heads=8, dropout=0.1)

    def forward(self, x, z):
        src_mask = self.make_src_mask(x)
        trg_mask = self.make_trg_mask(z)
        enc_src = self.encoder(x, src_mask)
        output = self.decoder(z, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len, device=trg.device)).type(torch.bool)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
