# server.py
import io
import base64
import logging
import traceback
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from film_unet import FiLM_UNet

# -----------------------------
# 기본 설정
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "film_unet_best.pth"

# uvicorn / Render 로그에 찍기 위한 로거
logger = logging.getLogger("uvicorn.error")

# -----------------------------
# 모델 로드 (서버 시작 시 1번만)
# -----------------------------
model = FiLM_UNet(user_dim=4)

try:
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    logger.info(f"✅ 모델 로드 완료: {MODEL_PATH} (device={DEVICE})")
except Exception as e:
    logger.error(f"❌ 모델 로드 실패: {e}")
    raise

model.to(DEVICE)
model.eval()

img_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop((256, 256)),
    T.ToTensor(),  # (C,H,W) 0~1
])


# -----------------------------
# 요청 바디 스키마
# -----------------------------
class CorrectionRequest(BaseModel):
    image: str              # base64 문자열
    user_vec: List[float]   # [p, d, t, deltaE]


# -----------------------------
# FastAPI 앱 + CORS
# -----------------------------
app = FastAPI()

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


# -----------------------------
# /correct 엔드포인트
# -----------------------------
@app.post("/correct")
def correct_color(req: CorrectionRequest):
    """
    입력:
      - image: base64 string (JPEG/PNG 등)
      - user_vec: [p, d, t, deltaE]

    출력:
      - {"corrected_image": "<base64 PNG>"}
    """
    try:
        # ---- user_vec 검증 ----
        if len(req.user_vec) != 4:
            raise HTTPException(
                status_code=400,
                detail=f"user_vec must be length 4, got {len(req.user_vec)}",
            )

        logger.info(f"📥 /correct called, user_vec={req.user_vec}")

        # ---- 1) base64 → PIL 변환 ----
        try:
            img_bytes = base64.b64decode(req.image)
        except Exception as e:
            logger.error("Base64 decode error: %s", e)
            raise HTTPException(status_code=400, detail=f"base64 decode error: {e}")

        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            # 아이폰 세로사진 회전 보정 + RGB 변환
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        except Exception as e:
            logger.error("PIL open/transpose error: %s", e)
            raise HTTPException(status_code=400, detail=f"PIL error: {e}")

        # ---- 2) 전처리 (256x256, Tensor) ----
        x = img_transform(pil_img).unsqueeze(0).to(DEVICE)  # (1,3,256,256)

        user_vec = torch.tensor(
            [req.user_vec], dtype=torch.float32, device=DEVICE
        )  # (1,4)

        # ---- 3) 모델 추론 ----
        with torch.no_grad():
            y = model(x, user_vec)   # (1,3,256,256), 0~1

            # 고주파 노이즈 줄이기
            y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)

            # 원본과 블렌딩 (너무 과하게 안 바뀌게)
            alpha = 0.6  # 0.0 = 원본 / 1.0 = 모델 결과
            y = alpha * y + (1.0 - alpha) * x

        # ---- 4) 이미지 후처리 + base64 인코딩 ----
        y = y.squeeze(0).cpu().clamp(0, 1)  # (3,256,256)
        out_pil = T.ToPILImage()(y)

        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        logger.info("✅ /correct success")
        return {"corrected_image": out_b64}

    except HTTPException:
        # 위에서 직접 던진 HTTPException은 그대로 전달
        raise
    except Exception as e:
        # 예기치 못한 에러는 로그 + 500으로 래핑
        tb = traceback.format_exc()
        logger.error("❌ /correct unexpected error: %s\n%s", e, tb)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# 로컬 테스트용 (Render에선 필요 X)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
