import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_embed=512, max_len=256):
        super(PositionalEncoding, self).__init__()

        # 위치(position) 벡터 (0부터 max_len-1까지)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 각 인덱스별 차원 i 계산 (0부터 d_embed-1까지)
        div_term = 10000**(torch.arange(0, d_embed, 2).float() / d_embed)
        
        PE = torch.zeros(max_len, d_embed)
        PE.requires_grad = False  # 위치 인코딩은 학습되지 않음
        
        PE[:, 0::2] = torch.sin(position / div_term)  # 짝수 인덱스에 sin 적용
        PE[:, 1::2] = torch.cos(position / div_term)  # 홀수 인덱스에 cos 적용
        
        self.PE = PE.unsqueeze(0)  # 배치 차원 추가

    def forward(self, x):
        _, seq_len, _ = x.size()
        PE = self.PE[:, :seq_len, :].to(x.device)  # 입력과 같은 길이로 자르고, 동일한 디바이스로 이동
        out = x + PE  # 입력에 위치 인코딩 추가
        return out
