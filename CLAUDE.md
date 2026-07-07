# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file CLI demo (`demo.py`) that screenshots the whole screen and runs PaddleOCR on it, printing recognized text + timings and saving an annotated image.

## Environment

- Runs in the conda env **`paddleocr`** (Python 3.11): `conda activate paddleocr`, or call the interpreter directly at `C:\Users\Ziqi\.conda\envs\paddleocr\python.exe`.
- **paddle must be 3.1+** (env has `paddlepaddle-gpu==3.3.1`, CUDA 12.6). paddle 3.0.0 fails with `strides is not right` against paddleocr 3.7 models.

## Run

```bash
python demo.py --device gpu        # GPU (~3s on full screen)
python demo.py --device cpu        # CPU (slow, ~2min on a 2560x1440 grab)
```

There are no automated tests; verification is running the demo and inspecting console output + the saved `result.jpg`.

## Non-obvious gotchas (already handled in demo.py)

- **CPU requires `enable_mkldnn=False`.** With oneDNN on, CPU inference crashes (`ConvertPirAttribute2RuntimeAttribute not support`). The code disables it only for `--device cpu`.
- **Console UTF-8.** `sys.stdout.reconfigure(encoding="utf-8")` is forced at startup, otherwise Chinese output is mojibake on Windows when piped/redirected.
- **mss returns BGRA**; the alpha channel is dropped to hand PaddleOCR a BGR array. `monitors[0]` is the all-screens bounding box; `monitors[1+]` are individual displays.
- The doc-orientation / unwarping / textline-orientation sub-modules are disabled in the `PaddleOCR(...)` call to speed up and avoid extra model downloads.
- **PaddleOCR HPI / OpenVINO is not available on native Windows.** Enabling `enable_hpi=True` requires `ultra-infer`, which PaddleX only ships prebuilt for Linux. We intentionally do not benchmark `paddle-openvino` here.

## Result parsing (PaddleOCR 3.x)

`ocr.predict(img)` returns a list; `result[0]` is a dict-like with `rec_texts` / `rec_scores`, and `.save_to_img(path)` writes the annotated visualization (renders CJK via PaddleOCR's bundled font — don't use raw `cv2.putText` for Chinese).
