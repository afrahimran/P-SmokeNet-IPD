import math
import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp] if self.chomp > 0 else x

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (k - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.out_relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.chomp1(y)
        y = self.relu1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = self.relu2(y)
        y = self.drop2(y)

        res = x if self.down is None else self.down(x)
        return self.out_relu(y + res)

class TCN(nn.Module):
    def __init__(self, input_dim, channels=(256, 256, 128), k=3, dropout=0.2):
        super().__init__()
        blocks = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            blocks.append(TCNBlock(in_ch, out_ch, k=k, dilation=dilation, dropout=dropout))
            in_ch = out_ch
        self.net = nn.ModuleList(blocks)

    def forward(self, x):
        for b in self.net:
            x = b(x)
        return x  # [B, C, T]

class SelfAttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, h):
        # h: [B, T, C]
        Q = self.query(h)  # [B,T,C]
        K = self.key(h)    # [B,T,C]
        V = self.value(h)  # [B,T,C]

        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(h.size(-1))  
        w = torch.softmax(scores, dim=-1)                                    
        ctx = torch.matmul(w, V)                                           
        z = ctx.mean(dim=1)                                                  
        return z

class SmokeFusionTCN(nn.Module):
    def __init__(self, seq_dim: int, fft_k: int, tcn_channels=(256, 256, 128), dropout=0.2):
        super().__init__()
        self.tcn = TCN(seq_dim, channels=tcn_channels, k=3, dropout=dropout)
        self.att = SelfAttentionPool(tcn_channels[-1])

        fusion_in = tcn_channels[-1] + fft_k
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, 1),
        )

    def forward(self, x_seq, x_fft):
        x = x_seq.transpose(1, 2)           
        h = self.tcn(x).transpose(1, 2)     
        z = self.att(h)                     
        zf = torch.cat([z, x_fft], dim=1)   
        logit = self.head(zf).squeeze(-1)   
        return logit