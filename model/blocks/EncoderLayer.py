import torch
import torch.nn as nn

from model.layers.MultiHeadAttention import MultiHeadAttention
from model.layers.PositionwiseFeedForward import PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self,  dff=2048, d_model=512, d_embed=512, n_head=8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.FFN = PositionwiseFeedForward(d_model=d_model, hidden=dff, drop_prob=dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):

        # 1. Multi-Head Attention
        out = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. Add & Norm
        out = self.dropout1(out)
        out = self.layer_norm1(out + x)

        # 3. Feed Forward
        x = out
        out = self.FFN(out)

        # 4. Add & Norm
        out = self.dropout2(out)
        out = self.layer_norm2(out + x)

        return out
