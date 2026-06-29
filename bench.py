"""CPU vs GPU 耗时对比（多轮 + 预热 + 取中位数，数据更稳）。

对同一张屏幕截图分别用两种设备运行：
  - 模型初始化计时一次
  - 先跑 1 轮预热（不计入），消除冷启动 / CUDA kernel 编译等一次性开销
  - 再跑 N 轮，取中位数 / 最小 / 最大

CPU 单轮推理很慢，默认轮数比 GPU 少，可用 --cpu-rounds / --gpu-rounds 调整。
"""

import argparse
import statistics
import sys
import time

import mss
import numpy as np

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from paddleocr import PaddleOCR


def grab_screen() -> np.ndarray:
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[0])
    return np.array(shot)[:, :, :3]


def bench(device: str, image: np.ndarray, rounds: int, warmup: int) -> dict:
    kwargs = dict(
        lang="ch",
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    if device == "cpu":
        kwargs["enable_mkldnn"] = False

    t0 = time.perf_counter()
    ocr = PaddleOCR(**kwargs)
    init_cost = time.perf_counter() - t0

    for _ in range(warmup):
        ocr.predict(image)

    samples = []
    lines = 0
    for i in range(rounds):
        t = time.perf_counter()
        result = ocr.predict(image)
        cost = time.perf_counter() - t
        samples.append(cost)
        lines = len(result[0].get("rec_texts", []))
        print(f"    第 {i + 1}/{rounds} 轮：{cost:.2f}s")

    return {
        "init": init_cost,
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "lines": lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU vs GPU OCR 耗时对比")
    parser.add_argument("--gpu-rounds", type=int, default=7, help="GPU 计时轮数（默认 7）")
    parser.add_argument("--cpu-rounds", type=int, default=3, help="CPU 计时轮数（默认 3）")
    parser.add_argument("--warmup", type=int, default=1, help="每种设备的预热轮数（默认 1，不计时）")
    args = parser.parse_args()

    print("截取屏幕（同一张图用于两种设备）……")
    image = grab_screen()
    print(f"图像尺寸：{image.shape[1]}x{image.shape[0]}\n")

    rounds_map = {"gpu": args.gpu_rounds, "cpu": args.cpu_rounds}
    results = {}
    for device in ("gpu", "cpu"):
        print(f"==== 测试 {device.upper()}（预热 {args.warmup} 轮 + 计时 {rounds_map[device]} 轮）====")
        results[device] = bench(device, image, rounds_map[device], args.warmup)
        r = results[device]
        print(f"  → 中位数 {r['median']:.2f}s（min {r['min']:.2f}s / max {r['max']:.2f}s）| {r['lines']} 行\n")

    g, c = results["gpu"], results["cpu"]
    print("==================== 对比汇总 ====================")
    print(f"{'指标':<14}{'GPU':>12}{'CPU':>12}{'CPU/GPU':>12}")
    print("-" * 50)
    rows = [
        ("模型初始化", "init"),
        ("推理(中位数)", "median"),
        ("推理(最快)", "min"),
        ("推理(最慢)", "max"),
    ]
    for label, key in rows:
        ratio = c[key] / g[key] if g[key] > 0 else float("inf")
        print(f"{label:<14}{g[key]:>11.2f}s{c[key]:>11.2f}s{ratio:>11.1f}x")
    print("=" * 50)


if __name__ == "__main__":
    main()
