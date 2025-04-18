import torch
import torch.nn as nn

class DenoiseAttentionModel(nn.Module):
    def __init__(self, input_dim=1, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(embed_dim, input_dim)

    def forward(self, x, src_key_padding_mask=None):
        # x: (batch, channel, time)
        x = x.permute(0, 2, 1)  # (batch, time, channel)
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # (time, batch, embed_dim)
        for attn in self.attention_layers:
            x = attn(x, src_key_padding_mask=src_key_padding_mask)
        x = x.permute(1, 0, 2)  # (batch, time, embed_dim)
        x = self.output_proj(x)
        x = x.permute(0, 2, 1)  # (batch, channel, time)
        return x