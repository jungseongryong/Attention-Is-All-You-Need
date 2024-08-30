import math
from torch import nn


class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None, e=1e22):
        """
        4차원의 텐서가 들어온다
        [batch_size, num_heads, 문장 길이, d_model/num_heads]
        """
        batch_size, head, seq_len, d_tensor = K.size()

        # 1.MatMul, Scale
        K_t = K.transpose(2,3) # [batch_size, num_heads, d_model/num_heads, 문장 길이]
        score = (Q @ K_t) / math.sqrt(d_tensor)

        # 2. Mask(opt.)
        if mask is not None:
            score -= (mask * e)

        # 3. SoftMax
        score = nn.Softmax(dim=-1)(score)

        #4. MatMul
        V = score @ V

        return V, score
