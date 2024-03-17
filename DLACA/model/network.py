import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import numpy as np


class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(start_dim=1),
        )

    def forward(self, inputs):
        x_fault = self.backbone(inputs)
        return x_fault


class CLS(nn.Module):
    def __init__(self, f_num):
        super(CLS, self).__init__()
        self.cls = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, f_num)
        )

    def forward(self, inputs):
        out = self.cls(inputs)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=.0):
        super().__init__()
        hidden_dim = dim * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        head_dim = int(dim / heads)
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_map = self.attend(dots)
        out = torch.matmul(attn_map, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out


class Transformer(nn.Module):
    def __init__(self, dim, num, depth=2, heads=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                Attention(dim=dim, heads=heads),
                FeedForward(dim),
                nn.LayerNorm(dim)
            ]))
        self.cls = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num)
        )

    def forward(self, x):
        x = x.unsqueeze(dim=0)
        for ln1, attn, ff, ln2 in self.layers:
            out = ln1(x)
            out = attn(out)
            x = out + x
            out = ln2(x)
            x = ff(out) + x
        feature = x.squeeze(dim=0)
        out = self.cls(feature)
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class TransEncode(nn.Module):
    def __init__(self, dim, depth=1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            Rearrange('b c n -> b n c'),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, 128))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim=128, heads=4)),
                PreNorm(dim, FeedForward(dim=128))
            ]))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 192)
        )

    def forward(self, x):
        out = self.embedding(x)
        out = out + self.pos_embedding
        for attn, ff in self.layers:
            out = attn(out) + out
            out = ff(out) + out

        out = out.mean(dim=1)
        out = self.mlp_head(out)
        return out


class TransCLS(nn.Module):
    def __init__(self, dim, num, depth=1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            Rearrange('b c n -> b n c'),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, 128))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim=128, heads=4)),
                PreNorm(dim, FeedForward(dim=128))
            ]))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, num)
        )

    def forward(self, x):
        out = self.embedding(x)
        out = out + self.pos_embedding
        for attn, ff in self.layers:
            out = attn(out) + out
            out = ff(out) + out

        out = out.mean(dim=1)
        out = self.mlp_head(out)
        return out
