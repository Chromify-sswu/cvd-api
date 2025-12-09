import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# FiLM Layer
# ============================================================

class FiLM(nn.Module):
    def __init__(self, channels, user_dim=4, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(user_dim, hidden)
        self.fc2 = nn.Linear(hidden, channels * 2)

    def forward(self, feat, user_vec):
        # user_vec: (B, user_dim)
        h = torch.relu(self.fc1(user_vec))   # (B, hidden)
        gamma, beta = self.fc2(h).chunk(2, dim=1)  # (B, channels), (B, channels)

        # reshape for broadcasting
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)

        return feat * gamma + beta


# ============================================================
# U-Net Basic Conv Block
# ============================================================

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(),
    )


# ============================================================
# FiLM U-Net (MODEL)
# ============================================================

class FiLM_UNet(nn.Module):
    def __init__(self, user_dim=4, base=32):
        super().__init__()

        # Encoder
        self.e1 = conv_block(3, base)
        self.p1 = nn.MaxPool2d(2)

        self.e2 = conv_block(base, base*2)
        self.p2 = nn.MaxPool2d(2)

        self.e3 = conv_block(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)

        # Bottleneck + FiLM
        self.bottleneck = conv_block(base*4, base*8)
        self.film = FiLM(base*8, user_dim=user_dim)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.d3 = conv_block(base*8, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = conv_block(base*4, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = conv_block(base*2, base)

        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x, user_vec):
        # Encoder
        e1 = self.e1(x)
        p1 = self.p1(e1)

        e2 = self.e2(p1)
        p2 = self.p2(e2)

        e3 = self.e3(p2)
        p3 = self.p3(e3)

        # Bottleneck + FiLM
        b = self.bottleneck(p3)
        b = self.film(b, user_vec)

        # Decoder
        u3 = self.up3(b)
        d3 = self.d3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.d2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.d1(torch.cat([u1, e1], dim=1))

        return torch.sigmoid(self.out(d1))