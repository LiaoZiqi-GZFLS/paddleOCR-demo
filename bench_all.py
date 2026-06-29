"""PaddleOCR vs EasyOCR，CPU / GPU 四种组合的速度对比（主程序）。

截一次屏并存成 PNG，所有组合用同一张图，最公平。每个组合在独立子进程中运行
（见 ocr_worker.py）。

注意：paddle 与 torch(easyocr) 的 cuDNN 在同一进程/环境内互斥，因此两个引擎分别使用
各自的 conda 环境解释器：
  - PaddleOCR  → 纯 paddle 环境 `pdl`
  - EasyOCR    → 环境 `paddleocr`（含 torch）
可用 --paddle-python / --easyocr-python 覆盖。
"""

import argparse
import json
import os
import subprocess
import sys

import mss
import numpy as np
from PIL import Image

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "ocr_worker.py")

DEFAULT_PADDLE_PY = r"C:\Users\Ziqi\.conda\envs\pdl\python.exe"
DEFAULT_EASYOCR_PY = r"C:\Users\Ziqi\.conda\envs\paddleocr\python.exe"


def grab_and_save(path: str) -> tuple:
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[0])
    arr = np.array(shot)[:, :, :3]  # BGR
    Image.fromarray(arr[:, :, ::-1]).save(path)  # 转 RGB 存 PNG
    return arr.shape[1], arr.shape[0]


def run_combo(python_exe: str, engine: str, device: str, image: str, rounds: int, warmup: int) -> dict | None:
    cmd = [python_exe, WORKER, "--engine", engine, "--device", device,
           "--image", image, "--rounds", str(rounds), "--warmup", str(warmup)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)
    for line in proc.stderr.splitlines():
        if "轮：" in line:
            print(line, flush=True)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
    print(f"  [失败] {engine}-{device}:\n{tail}\n", flush=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="PaddleOCR vs EasyOCR CPU/GPU 速度对比")
    parser.add_argument("--gpu-rounds", type=int, default=5, help="GPU 计时轮数（默认 5）")
    parser.add_argument("--cpu-rounds", type=int, default=2, help="CPU 计时轮数（默认 2）")
    parser.add_argument("--warmup", type=int, default=1, help="每组合预热轮数（默认 1）")
    parser.add_argument("--only", nargs="+",
                        default=["paddle-gpu", "easyocr-gpu", "paddle-cpu", "easyocr-cpu"],
                        help="只跑指定组合")
    parser.add_argument("--image", default=os.path.join(HERE, "bench_shot.png"),
                        help="截图保存/复用路径")
    parser.add_argument("--paddle-python", default=DEFAULT_PADDLE_PY)
    parser.add_argument("--easyocr-python", default=DEFAULT_EASYOCR_PY)
    args = parser.parse_args()

    py_map = {"paddle": args.paddle_python, "easyocr": args.easyocr_python}

    w, h = grab_and_save(args.image)
    print(f"截图已保存：{args.image}（{w}x{h}），所有组合共用此图\n")

    rounds_map = {"gpu": args.gpu_rounds, "cpu": args.cpu_rounds}
    results = {}
    for combo in args.only:
        engine, device = combo.split("-")
        print(f"==== {combo}（预热 {args.warmup} + 计时 {rounds_map[device]} 轮）====", flush=True)
        results[combo] = run_combo(py_map[engine], engine, device, args.image,
                                   rounds_map[device], args.warmup)
        r = results[combo]
        if r:
            print(f"  → 初始化 {r['init']:.2f}s | 推理中位数 {r['median']:.2f}s | {r['lines']} 行\n", flush=True)

    print("=" * 66)
    print(f"{'组合':<16}{'初始化':>12}{'推理(中位)':>14}{'最快':>10}{'行数':>8}")
    print("-" * 66)
    for combo in args.only:
        r = results.get(combo)
        if r is None:
            print(f"{combo:<16}{'—— 失败 ——':>24}")
            continue
        print(f"{combo:<16}{r['init']:>11.2f}s{r['median']:>13.2f}s{r['min']:>9.2f}s{r['lines']:>8}")
    print("=" * 66)


if __name__ == "__main__":
    main()
