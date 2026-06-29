# paddleOCR-demo

对**整个屏幕截图**并用 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 识别其中文字的命令行 Demo。支持 CPU / GPU 切换，识别中英文混合文本，打印识别结果与耗时，并保存带检测框的可视化图片。

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

## 速度对比：PaddleOCR vs EasyOCR，CPU vs GPU

同一张 2560×1440 全屏截图，预热 1 轮后取多轮推理中位数（GPU 5 轮 / CPU 2 轮）。测试机为 CUDA 12.6 GPU。

| 组合 | 模型初始化 | 推理(中位) | 最快 | 识别行数 |
|------|-----------|-----------|------|---------|
| **paddle-gpu** | 5.47s | **2.56s** | 2.54s | 64 |
| easyocr-gpu | 2.48s | 7.18s | 7.04s | 164 |
| paddle-cpu | 7.25s | 195.36s | 190.26s | 64 |
| easyocr-cpu | 2.36s | 19.68s | 19.66s | 164 |

**结论：**

- **GPU 上 PaddleOCR 最快**：2.56s，约为 EasyOCR（7.18s）的 1/2.8。
- **CPU 上反过来 EasyOCR 大胜**：19.68s，而 PaddleOCR 慢到 195s（≈10×），大图基本不可用。
- **GPU 加速比**：PaddleOCR 约 76×，EasyOCR 约 2.7×——Paddle 吃 GPU 的收益远大于 EasyOCR。
- **行数差异**（64 vs 164）非准确率差异，而是切分粒度不同：EasyOCR 倾向碎框、Paddle 倾向整行合并。

**选型建议：有 GPU 选 PaddleOCR，只有 CPU 选 EasyOCR。**

### 复现

PaddleOCR 与 EasyOCR 不能共用一个环境——`paddlex` 会自动导入 `torch`，其 cuDNN 与 paddle 的 cuDNN 在同一进程内冲突（Windows 下报 `WinError 127`）。因此用**两个独立 conda 环境 + 子进程隔离**：

```bash
# 环境 1：纯 paddle（不要装 torch），跑 PaddleOCR
conda create -n pdl python=3.11 -y
C:\Users\<you>\.conda\envs\pdl\python.exe -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
C:\Users\<you>\.conda\envs\pdl\python.exe -m pip install paddleocr mss pillow numpy

# 环境 2：含 torch(cu126) 的环境，跑 EasyOCR
conda activate paddleocr
pip install easyocr torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 运行四路对比（按引擎自动选对应解释器，可用 --paddle-python / --easyocr-python 覆盖）
python bench_all.py --gpu-rounds 5 --cpu-rounds 2
```

`bench_all.py` 截一次屏存为 `bench_shot.png`，四个组合共用同一张图保证公平；每个组合在独立子进程（`ocr_worker.py`）中运行。仅对比 PaddleOCR 的 CPU/GPU 可直接用 `python bench.py`（单环境即可）。
