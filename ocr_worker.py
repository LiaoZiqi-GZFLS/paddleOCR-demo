"""单组合 OCR 计时 worker（被 bench_all.py 以子进程方式调用）。

独立进程运行，避免 paddle 与 torch(easyocr) 在同一进程内 cuDNN 冲突。
从磁盘读取固定图片，按指定引擎 / 设备运行，结果以 RESULT_JSON: 前缀打印到 stdout。
"""

import argparse
import json
import statistics
import sys
import time

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["paddle", "easyocr"])
    ap.add_argument("--device", required=True, choices=["cpu", "gpu"])
    ap.add_argument("--image", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    # 用 PIL 读图（不引入 cv2，避免其加载的 DLL 干扰 paddle 的 cuDNN 依赖解析）
    img_rgb = np.array(Image.open(args.image).convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1]

    if args.engine == "paddle":
        from paddleocr import PaddleOCR

        kwargs = dict(
            lang="ch",
            device=args.device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        if args.device == "cpu":
            kwargs["enable_mkldnn"] = False
        t0 = time.perf_counter()
        ocr = PaddleOCR(**kwargs)
        init = time.perf_counter() - t0
        infer = lambda: ocr.predict(img_bgr)
        count = lambda r: len(r[0].get("rec_texts", []))
    else:
        import easyocr

        t0 = time.perf_counter()
        reader = easyocr.Reader(["ch_sim", "en"], gpu=(args.device == "gpu"))
        init = time.perf_counter() - t0
        infer = lambda: reader.readtext(img_rgb)
        count = lambda r: len(r)

    for _ in range(args.warmup):
        infer()

    samples, lines = [], 0
    for i in range(args.rounds):
        t = time.perf_counter()
        result = infer()
        cost = time.perf_counter() - t
        samples.append(cost)
        lines = count(result)
        print(f"  [{args.engine}-{args.device}] 第 {i + 1}/{args.rounds} 轮：{cost:.2f}s", file=sys.stderr, flush=True)

    out = {
        "init": init,
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "lines": lines,
    }
    print("RESULT_JSON:" + json.dumps(out))


if __name__ == "__main__":
    main()
