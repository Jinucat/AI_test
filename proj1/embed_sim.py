import argparse
import itertools
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# ---- MediaPipe ----
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imread_unicode(path: Path):
    """
    Windows 한글/공백 경로에서도 안전하게 읽기 (cv2.imread 대체)
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def resize_for_preview(image):
    h, w = image.shape[:2]
    if h < w:
        new_h = max(1, math.floor(h / (w / DESIRED_WIDTH)))
        img = cv2.resize(image, (DESIRED_WIDTH, new_h))
    else:
        new_w = max(1, math.floor(w / (h / DESIRED_HEIGHT)))
        img = cv2.resize(image, (new_w, DESIRED_HEIGHT))
    return img


def collect_images(images, directory):
    files = []
    if images:
        for p in images:
            path = Path(p)
            if path.is_file():
                files.append(path)
            else:
                print(f"[WARN] 파일을 찾을 수 없음: {path}")
    if directory:
        d = Path(directory)
        if not d.exists():
            print(f"[WARN] 디렉터리를 찾을 수 없음: {d}")
        else:
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    files.append(p)

    # 중복 제거 & 존재 확인
    uniq = []
    seen = set()
    for f in files:
        key = f.resolve()
        if key not in seen and f.exists():
            uniq.append(f)
            seen.add(key)
    return uniq


def build_embedder(model_path, l2_normalize=True, quantize=True):
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.ImageEmbedderOptions(
        base_options=base_options,
        l2_normalize=l2_normalize,
        quantize=quantize
    )
    return vision.ImageEmbedder.create_from_options(options)


def make_mp_image(path: Path):
    # mediapipe는 파일 경로에서 바로 읽을 수 있으나, 한글 경로 이슈가 있을 수 있어
    # 안전하게 임시 디코드 → imencode → create_from_file 로 우회 가능.
    # 여기선 경로 직접 전달(일반적으로 동작). 문제 시 아래 주석된 대안 사용.
    return mp.Image.create_from_file(str(path))

    # 대안:
    # img = imread_unicode(path)
    # if img is None:
    #     raise ValueError(f"이미지 로드 실패: {path}")
    # # mediapipe는 ndarray로 직접 받는 API가 공식적으론 없음.
    # tmp = cv2.imencode('.png', img)[1].tobytes()
    # return mp.Image(image_format=mp.ImageFormat.SRGB, data=tmp)


def compute_similarities(embedder, image_paths):
    mp_images = [make_mp_image(p) for p in image_paths]
    emb_results = [embedder.embed(im).embeddings[0] for im in mp_images]

    def sim(i, j):
        return vision.ImageEmbedder.cosine_similarity(emb_results[i], emb_results[j])

    n = len(image_paths)
    sims = np.eye(n, dtype=float)
    for i, j in itertools.combinations(range(n), 2):
        s = sim(i, j)
        sims[i, j] = sims[j, i] = s
    return sims


def preview_images(paths, window_prefix="preview"):
    for p in paths:
        img = imread_unicode(p)
        if img is None:
            print(f"[WARN] 미리보기 실패(읽기 실패): {p}")
            continue
        show = resize_for_preview(img)
        cv2.imshow(f"{window_prefix}: {p.name}", show)
    if paths:
        print("[INFO] 미리보기 창이 열렸습니다. 아무 키나 누르면 닫힙니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="MediaPipe Image Embedding 유사도 테스트")
    ap.add_argument("--model", required=True, help="TFLite 모델 경로 (예: models/mobilenet_v3_small.tflite)")
    ap.add_argument("--images", nargs="*", help="이미지 파일 경로들(공백으로 구분)")
    ap.add_argument("--dir", help="이미지 폴더 경로 (jpg/png/webp/… 자동 수집)")
    ap.add_argument("--no-gui", action="store_true", help="미리보기 창 표시 안 함")
    ap.add_argument("--no-l2", action="store_true", help="L2 normalize 끔")
    ap.add_argument("--no-quant", action="store_true", help="quantize 끔")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] 모델 파일이 없습니다: {model_path}")
        sys.exit(1)

    files = collect_images(args.images, args.dir)
    if len(files) < 2:
        print("[ERROR] 적어도 2장의 이미지를 지정하세요. 예: --images a.jpg b.jpg  또는 --dir images")
        sys.exit(1)

    if not args.no_gui:
        preview_images(files)

    print(f"[INFO] 이미지 개수: {len(files)}")
    for i, p in enumerate(files):
        print(f"  [{i}] {p}")

    # Embedder 생성
    embedder = build_embedder(
        model_path=model_path,
        l2_normalize=not args.no_l2,
        quantize=not args.no_quant
    )

    sims = compute_similarities(embedder, files)

    # 결과 출력
    names = [p.name for p in files]
    w = max(len(n) for n in names) + 2
    header = " " * w + " ".join(f"{n:>8}" for n in range(len(names)))
    print("\n[유사도 행렬: cosine_similarity]\n")
    print(header)
    for i, row in enumerate(sims):
        print(f"{str(i).rjust(w-2)}  " + " ".join(f"{v:>8.4f}" for v in row))

    if len(files) == 2:
        persent = float(sims[0, 1])        # 0.9154 같은 실수
        percent_str = f"{persent * 100:.0f}%"
        print(f"\n[결과] {names[0]}  vs  {names[1]}  →  similarity = {percent_str}")
    else:
        # 가장 비슷한 쌍 TOP-3
        pairs = []
        for i, j in itertools.combinations(range(len(files)), 2):
            pairs.append(((i, j), sims[i, j]))
        pairs.sort(key=lambda x: x[1], reverse=True)
        print("\n[가장 유사한 쌍 TOP-3]")
        for (i, j), s in pairs[:3]:
            print(f"  ({i}) {names[i]}  ↔  ({j}) {names[j]}  :  {s:.4f}")


if __name__ == "__main__":
    main()