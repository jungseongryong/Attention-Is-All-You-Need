import torch
import torch.nn as nn

from model.layers.MultiHeadAttention import MultiHeadAttention
from model.layers.PositionwiseFeedForward import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self,  dff=2048, d_model=512, d_embed=512, n_head=8, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.attention2 = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.FFN = PositionwiseFeedForward(d_model=d_model, hidden=dff, drop_prob=dropout)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x, c, trg_mask, src_mask):

        # 1. Masked Multi-Head Attention
        out = self.attention1(q=x, k=x, v=x, mask=trg_mask)

        # 2. Add & Norm
        out = self.dropout1(out)
        out = self.layer_norm1(out + x)

        # 3. Multi-Head Attention[Decoder-Encoder attention]
        x = out
        out = self.attention2(q=out, k=c, v=c, mask=src_mask)

        # 4. Add & Norm
        out = self.dropout2(out)
        out = self.layer_norm2(out + x)

        # 5. Feed Forward
        x = out
        out = self.FFN(out)

        # 6. Add & Norm
        out = self.dropout3(out)
        out = self.layer_norm3(out + x)

        return out
