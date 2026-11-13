from pathlib import Path
from PIL import Image
import numpy as np

# 실제 데이터셋 폴더 경로로 수정
DATA_DIR = Path("dataset")

widths = []
heights = []

for img_path in DATA_DIR.rglob("*"):

    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        continue

    try:
        with Image.open(img_path) as img:
            width, height = img.size

            widths.append(width)
            heights.append(height)

    except Exception as e:
        print(f"이미지 읽기 실패: {img_path}, {e}")

widths = np.array(widths)
heights = np.array(heights)

print("===== 이미지 크기 분석 =====")
print(f"전체 이미지 수: {len(widths)}")

if len(widths) > 0:
    print(f"평균 너비: {widths.mean():.2f}px")
    print(f"평균 높이: {heights.mean():.2f}px")

    print(f"중앙값 너비: {np.median(widths):.0f}px")
    print(f"중앙값 높이: {np.median(heights):.0f}px")

    print(f"최소 크기: {widths.min()} × {heights.min()}")
    print(f"최대 크기: {widths.max()} × {heights.max()}")

    print(f"128×128 미만 이미지 수(너비 또는 높이): {np.sum((widths < 128) | (heights < 128))}")