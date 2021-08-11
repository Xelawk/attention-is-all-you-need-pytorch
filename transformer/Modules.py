import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # [b, n_heads, len, len]

        if mask is not None: # encoder: [b, 1, 1, len], decoder: [b, 1, len, len]
            # mask不为空是就是论文提到的 masked multi-head attention了，mask处要置为负无穷
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))  # trick
        output = torch.matmul(attn, v)

        return output, attn
