"""이미지 로딩: EXIF 방향 정규화, Alpha 합성, 다중 프레임(TIFF/GIF) 분리, HEIC 지원.

문서 "파일 유형별 입력 처리" 표와 "이미지 전처리 > 정규화" 행을 구현한다.
지오메트리 보정(Deskew 등)·명암 보정은 Stage 2 범위이므로 여기서는 하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, ImageSequence

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    _HEIF_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _HEIF_AVAILABLE = False

MULTI_FRAME_EXTENSIONS = {".tif", ".tiff", ".gif"}
HEIF_EXTENSIONS = {".heic", ".heif"}


@dataclass
class ImageFrame:
    index: int  # 1-based (다중 페이지/프레임 순서 유지)
    image: Image.Image  # RGB, EXIF 방향 적용 완료


def load_image_frames(path: str | Path) -> list[ImageFrame]:
    suffix = Path(path).suffix.lower()
    if suffix in HEIF_EXTENSIONS and not _HEIF_AVAILABLE:
        raise RuntimeError(
            "HEIC/HEIF 파일을 열려면 pillow-heif가 필요합니다 (pip install pillow-heif)."
        )

    img = Image.open(path)

    if suffix in MULTI_FRAME_EXTENSIONS:
        frames = [
            ImageFrame(index=i + 1, image=_normalize(frame.copy()))
            for i, frame in enumerate(ImageSequence.Iterator(img))
        ]
        return frames

    return [ImageFrame(index=1, image=_normalize(img))]


def _normalize(img: Image.Image) -> Image.Image:
    # EXIF 방향 적용 (전처리 "정규화": EXIF 회전).
    img = ImageOps.exif_transpose(img)

    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        # Alpha 합성: 흰 배경에 합성 (전처리 "정규화": Alpha 합성).
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background

    return img.convert("RGB")
