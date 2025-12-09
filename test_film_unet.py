import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


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


# ============================================================
# 모델 로드
# ============================================================

def load_model(ckpt_path, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = FiLM_UNet(user_dim=4, base=32).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # DDP 저장형식 대응
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    model.load_state_dict(new_state, strict=True)
    model.eval()
    return model, device


# ============================================================
# 더미 테스트 (선택)
# ============================================================

def test_dummy_forward(ckpt_path):
    model, device = load_model(ckpt_path)

    x = torch.rand(1, 3, 256, 256, device=device)
    user_vec = torch.tensor([[0.5, 0.8, 1.0, 0.0]], device=device)

    with torch.no_grad():
        y = model(x, user_vec)

    print("input :", x.shape)
    print("output:", y.shape)
    print("output range:", float(y.min()), float(y.max()))


# ============================================================
# 실제 이미지에 적용
# ============================================================

def run_on_image(
    ckpt_path,
    img_path,
    save_path="output.png",
    user_vec_list=[0.5, 0.8, 1.0, 0.0],
    image_size=256,
):
    model, device = load_model(ckpt_path)

    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),           # [0,1]
    ])
    x = transform(img).unsqueeze(0).to(device)  # (1,3,H,W)

    user_vec = torch.tensor([user_vec_list], dtype=torch.float32, device=device)

    with torch.no_grad():
        y = model(x, user_vec)

    y = y.squeeze(0).cpu()  # (3,H,W)
    out_img = T.ToPILImage()(y)
    out_img.save(save_path)
    print(f"Saved to: {save_path}")


# ============================================================
# main: CLI 인자 받기
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="film_unet_best.pth")
    parser.add_argument("--input", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--size", type=int, default=256)

    # user_vec도 CLI로 살짝 조절할 수 있게 (4차원이라고 가정)
    parser.add_argument("--u0", type=float, default=0.5)
    parser.add_argument("--u1", type=float, default=0.8)
    parser.add_argument("--u2", type=float, default=1.0)
    parser.add_argument("--u3", type=float, default=0.0)

    args = parser.parse_args()

    user_vec = [args.u0, args.u1, args.u2, args.u3]

    # 1) 더미 테스트 한 번 해보고 싶으면 주석 풀기
    # test_dummy_forward(args.ckpt)

    # 2) 실제 이미지에 적용
    run_on_image(
        ckpt_path=args.ckpt,
        img_path=args.input,
        save_path=args.output,
        user_vec_list=user_vec,
        image_size=args.size,
    )
