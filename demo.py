"""PaddleOCR 全屏截图识别 Demo。

对整个屏幕截图，用 PaddleOCR 识别其中的文字，打印结果并保存带标注的可视化图片，
同时统计模型初始化与推理的耗时。支持通过 --device 选择 CPU 或 GPU。

用法示例：
    python demo.py                 # 默认 CPU，中英文
    python demo.py --device gpu    # 使用 GPU
    python demo.py --device gpu --lang en --output out.jpg
"""

import argparse
import sys
import time

import mss
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaddleOCR 全屏截图识别 Demo")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="推理设备：cpu 或 gpu（默认 cpu）",
    )
    parser.add_argument(
        "--lang",
        default="ch",
        help="识别语言，ch=中英文混合，en=英文（默认 ch）",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=0,
        help="截图显示器编号：0=全部屏幕拼合，1=主屏，2=副屏……（默认 0）",
    )
    parser.add_argument(
        "--output",
        default="result.jpg",
        help="可视化结果保存路径（默认 result.jpg）",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    """校验设备可用性，GPU 不可用时给出清晰提示并退出。"""
    if device == "gpu":
        import paddle

        if not paddle.is_compiled_with_cuda():
            sys.exit("[错误] 当前 paddle 不是 GPU 版本，请安装 paddlepaddle-gpu 后再用 --device gpu")
        if paddle.device.cuda.device_count() == 0:
            sys.exit("[错误] 未检测到可用的 GPU，请检查显卡驱动 / CUDA 环境")
    return device


def grab_screen(monitor_index: int) -> np.ndarray:
    """对指定显示器截图，返回 BGR 格式的 numpy 数组（供 OpenCV / PaddleOCR 使用）。"""
    with mss.mss() as sct:
        if monitor_index < 0 or monitor_index >= len(sct.monitors):
            sys.exit(f"[错误] 显示器编号 {monitor_index} 无效，可用范围 0~{len(sct.monitors) - 1}")
        shot = sct.grab(sct.monitors[monitor_index])
    # mss 返回 BGRA，丢掉 alpha 通道得到 BGR
    return np.array(shot)[:, :, :3]


def main() -> None:
    # 强制 UTF-8 输出，避免 Windows 控制台 / 管道下中文乱码
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = parse_args()
    device = resolve_device(args.device)

    print(f"[1/4] 截取屏幕（monitor={args.monitor}）……")
    screen = grab_screen(args.monitor)
    print(f"      截图尺寸：{screen.shape[1]}x{screen.shape[0]}")

    print(f"[2/4] 初始化 PaddleOCR（device={device}, lang={args.lang}）……")
    from paddleocr import PaddleOCR

    t0 = time.perf_counter()
    ocr_kwargs = dict(
        lang=args.lang,
        device=device,
        # 关闭文档方向/扭曲校正等附加模块，加快速度、避免额外下载
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    if device == "cpu":
        # CPU 下 oneDNN(mkldnn) 加速与当前 paddle 版本存在兼容问题，关闭以保证可用
        ocr_kwargs["enable_mkldnn"] = False
    ocr = PaddleOCR(**ocr_kwargs)
    init_cost = time.perf_counter() - t0
    print(f"      模型初始化耗时：{init_cost:.2f} 秒")

    print("[3/4] 执行 OCR 推理……")
    t1 = time.perf_counter()
    result = ocr.predict(screen)
    infer_cost = time.perf_counter() - t1

    res = result[0]
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])

    print("\n========== 识别结果 ==========")
    for i, (text, score) in enumerate(zip(texts, scores), 1):
        print(f"{i:>3}. [{score:.2f}] {text}")
    print("==============================")
    print(f"共检测到 {len(texts)} 行文字")
    print(f"推理耗时：{infer_cost:.2f} 秒（模型初始化另计 {init_cost:.2f} 秒）")

    print(f"\n[4/4] 保存可视化结果到 {args.output} ……")
    res.save_to_img(args.output)
    print("完成。")


if __name__ == "__main__":
    main()
