# paddleOCR-demo

对**整个屏幕截图**并用 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 识别其中文字的命令行 Demo。支持 CPU / GPU 切换，识别中英文混合文本，打印识别结果与耗时，并保存带检测框的可视化图片。

另外提供 **PaddleOCR / EasyOCR / RapidOCR** 的 CPU/GPU 速度对比工具（`bench_all.py`）。

## 环境准备

已在 conda + Python 3.11 下验证。建议用独立环境：

```bash
conda create -n paddleocr python=3.11 -y
conda activate paddleocr

# GPU 版 paddle（CUDA 12.6，向后兼容更新的驱动）
pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
# 或纯 CPU：pip install paddlepaddle==3.3.1

pip install paddleocr mss opencv-python
```

> 注意：paddle 需 **3.1+**，3.0.0 与 paddleocr 3.7 的模型不兼容（会报 `strides is not right`）。

## 用法

```bash
python demo.py                          # 默认 CPU，中英文
python demo.py --device gpu             # 使用 GPU
python demo.py --device gpu --lang en   # 仅识别英文
python demo.py --monitor 1 --output out.jpg   # 只截主屏，自定义输出
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--device` | `cpu` 或 `gpu` | `cpu` |
| `--lang` | `ch`=中英文，`en`=英文 | `ch` |
| `--monitor` | `0`=全部屏幕拼合，`1`=主屏，`2`=副屏… | `0` |
| `--output` | 可视化结果保存路径 | `result.jpg` |

## 输出

- 控制台：逐行打印识别文字 + 置信度，末尾汇总检测行数、推理耗时、模型初始化耗时
- 图片：在截图上绘制检测框与识别文字（中文用 PaddleOCR 内置字体渲染），保存到 `--output`

首次运行会自动下载检测 / 识别模型。整屏大图在 CPU 上推理较慢（可达 1~2 分钟），GPU 通常几秒内完成。

## 速度对比：PaddleOCR vs EasyOCR vs RapidOCR，CPU vs GPU

同一张 2560×1440 全屏截图，预热 1 轮后取多轮推理中位数（GPU 5 轮 / CPU 2 轮）。测试机为 CUDA 12.6 GPU。

| 组合 | 模型初始化 | 推理(中位) | 最快 | 识别行数 |
|------|-----------|-----------|------|---------|
| **rapidocr-gpu** | 0.47s | **1.10s** | 0.63s | 124 |
| **paddle-gpu** | 5.63s | 4.90s | 4.87s | 145 |
| easyocr-gpu | 2.55s | 7.97s | 7.84s | 172 |
| rapidocr-cpu | 0.25s | 4.26s | 4.24s | 124 |
| easyocr-cpu | 2.29s | 23.10s | 23.05s | 172 |
| paddle-cpu | 4.97s | 265.17s | 264.72s | 145 |

**结论：**

- **RapidOCR 在 GPU 和 CPU 上都最快**：
  - GPU：1.10s，约为 PaddleOCR（4.90s）的 1/4.5、EasyOCR（7.97s）的 1/7.2。
  - CPU：4.26s，比 EasyOCR（23.10s）快 5.4 倍，比 PaddleOCR（265.17s）快 62 倍。
- **PaddleOCR 仍严重依赖 GPU**：CPU 265s，基本不可用；GPU 4.90s 尚可。
- **EasyOCR 比较均衡**：CPU 23s 可接受，GPU 8s 较慢。
- **行数差异**（124 / 145 / 172）非准确率差异，而是切分粒度不同：EasyOCR 最碎、Paddle 次之、Rapid 更倾向整行合并。

**选型建议：**
- **追求速度 / 无 GPU 也想快** → **RapidOCR**。
- **需要 PaddleOCR 生态 / 特定模型精度** → 有 GPU 选 PaddleOCR，无 GPU 避免使用。
- **不想折腾多环境 / 只要能用** → EasyOCR 单环境即可。

### 复现

PaddleOCR 与 EasyOCR/RapidOCR 不能共用一个环境——`paddlex` 会自动导入 `torch`，其 cuDNN 与 paddle 的 cuDNN 在同一进程内冲突（Windows 下报 `WinError 127`）。因此用**独立 conda 环境 + 子进程隔离**：

```bash
# 环境 1：纯 paddle（不要装 torch），跑 PaddleOCR
conda create -n pdl python=3.11 -y
C:\Users\<you>\.conda\envs\pdl\python.exe -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
C:\Users\<you>\.conda\envs\pdl\python.exe -m pip install paddleocr mss pillow numpy

# 环境 2：含 torch(cu126) 的环境，跑 EasyOCR 和 RapidOCR
conda create -n paddleocr python=3.11 -y
C:\Users\<you>\.conda\envs\paddleocr\python.exe -m pip install easyocr torch torchvision --index-url https://download.pytorch.org/whl/cu126
C:\Users\<you>\.conda\envs\paddleocr\python.exe -m pip install rapidocr-onnxruntime mss pillow numpy
# RapidOCR GPU 需要 onnxruntime-gpu（注意：>1.20 在部分环境有 CUDA 兼容问题，这里用 1.20.1）
C:\Users\<you>\.conda\envs\paddleocr\python.exe -m pip uninstall -y onnxruntime
C:\Users\<you>\.conda\envs\paddleocr\python.exe -m pip install onnxruntime-gpu==1.20.1

# 运行六路对比（按引擎自动选对应解释器，可用 --paddle-python / --easyocr-python / --rapidocr-python 覆盖）
python bench_all.py --gpu-rounds 5 --cpu-rounds 2
```

`bench_all.py` 截一次屏存为 `bench_shot.png`，六个组合共用同一张图保证公平；每个组合在独立子进程（`ocr_worker.py`）中运行。仅对比 PaddleOCR 的 CPU/GPU 可直接用 `python bench.py`（单环境即可）。

### 关于 OpenVINO / HPI 加速 CPU 的尝试

PaddleOCR 3.x 理论上支持 `enable_hpi=True` 启用高性能推理（HPI），让模型在 Intel CPU 上走 **OpenVINO** 后端以加速 CPU 推理。实际测试发现：

- Windows 下启用 `enable_hpi=True` 会报错 `Engine 'hpi' is unavailable because dependency 'ultra-infer' is not installed.`
- `ultra-infer`（PaddleX HPI 插件）目前**没有 Windows 官方预编译包**；PaddleX 文档明确建议在 Windows 上通过 Docker 或 WSL2 使用 HPI。

因此**本仓库未纳入 `paddle-openvino` 对比**。如果你想在 Windows 本地做 OpenVINO 加速，可选路线：

1. **在 WSL2/Docker（Linux）中跑**：安装 `paddlex --install hpi-cpu` 后即可用 `enable_hpi=True`，这是最省事的方案。
2. **导出 ONNX + OpenVINO Runtime / ONNX Runtime(OpenVINO EP)**：工作量较大，需要把 PaddleOCR 的 det/rec 模型导出为 ONNX，并手写前后处理。

由于当前目标只是本地快速对比，暂保留 6 路对比，不再继续 OpenVINO 分支。
