from torch import nn

from model.layers.ScaleDotProductAttention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_embed=512, d_model=512, start=None):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        if start is not None:
            self.w_q = nn.Linear(d_embed, d_model)
            self.w_k = nn.Linear(d_embed, d_model)
            self.w_v = nn.Linear(d_embed, d_model)
        else:
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. Linear
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. Split
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. Scaled Dot-Produt Attention
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. Concat, Linear
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization
        return out

    def split(self, tensor):
        """
        [batch_size, length, d_model] -> [batch_size, head, length, d_tensor]
        """
        batch_size, seq_len, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, seq_len, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        [batch_size, head, length, d_tensor] ->[batch_size, length, d_model]
        """
        batch_size, head, seq_len, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return tensor
