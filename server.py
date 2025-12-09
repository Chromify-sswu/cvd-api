# server.py
import io
import base64
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from film_unet import FiLM_UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "film_unet_best.pth"

# ---- 모델 로드 (서버 시작 시 1번만) ----
model = FiLM_UNet(user_dim=4)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

img_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop((256, 256)),
    T.ToTensor(),  # (C,H,W) 0~1
])


class CorrectionRequest(BaseModel):
    image: str          # base64 문자열
    user_vec: List[float]   # [p, d, t, deltaE]


app = FastAPI()

# ---- CORS (앱/웹에서 호출하기 쉽게) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 개발 단계: 전체 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.post("/correct")
def correct_color(req: CorrectionRequest):
    if len(req.user_vec) != 4:
        raise HTTPException(
            status_code=400,
            detail=f"user_vec must be length 4, got {len(req.user_vec)}",
        )

    # 1) base64 → PIL (회전 보정 포함)
    img_bytes = base64.b64decode(req.image)
    pil_img = Image.open(io.BytesIO(img_bytes))
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")

    # 2) 전처리
    x = img_transform(pil_img).unsqueeze(0).to(DEVICE)  # (1,3,256,256)

    user_vec = torch.tensor(
        [req.user_vec], dtype=torch.float32, device=DEVICE
    )  # (1,4)

    with torch.no_grad():
        # 3) 모델 추론
        y = model(x, user_vec)   # (1,3,256,256), 0~1

        # 🔧 고주파 노이즈 줄이기
        y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)

        # 🔧 원본과 블렌딩해서 덜 깨져 보이게
        alpha = 0.6  # 0.0 = 원본 / 1.0 = 모델 그 자체
        y = alpha * y + (1.0 - alpha) * x

    y = y.squeeze(0).cpu().clamp(0, 1)

    out_pil = T.ToPILImage()(y)

    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"corrected_image": out_b64}


# 로컬 테스트용 (Render에서는 안 써도 됨)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
