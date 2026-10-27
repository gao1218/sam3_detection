# SAM 3 图像分割工具

一个基于 Streamlit 和 Meta SAM 3 的本地图像分割应用，支持文字、交互式点和交互式框提示。点提示可将多个前景点分组到同一目标；框提示支持包含框和屏蔽框。结果可以查看、合并并下载为 PNG。

## 功能

- **文字提示**：输入对象描述，分割图像中符合描述的对象。
- **点提示**：添加前景点和背景点。可选择每个前景点代表一个目标，或手动将多个前景点分组到指定目标。
- **框提示**：绘制包含框和屏蔽框；绘制模式用于新增，移动/调整模式用于修改已有框。
- **Mask 结果**：查看叠加结果和独立二值 mask；可分别保存、合并为二值并集，或保存包含目标编号的标签 mask。
- **GPU 推理**：当 CUDA 可用时优先使用 GPU 和自动混合精度；也可通过环境变量切换到 CPU。

## 环境要求

### 推荐环境

- Windows 10/11 64 位。
- Python 3.12（推荐，与 SAM 3 上游依赖兼容性较好）。SAM 3 上游要求 Python 3.12 或更高版本。
- NVIDIA CUDA 显卡及兼容的 NVIDIA 驱动。应用的已验证配置为 CUDA 13.0 版 PyTorch；显存建议至少 6 GB。图像分辨率、提示数量和其他 GPU 程序会影响实际显存需求。
- 建议至少 16 GB 系统内存，并预留充足的 Windows 页面文件。SAM 3 权重较大，加载时可能出现较高内存和磁盘读写压力。
- 数 GB 的可用磁盘空间，用于 Python 环境、SAM 3 权重和运行缓存。

SAM 3 上游的基础要求是 PyTorch 2.7+ 和 CUDA 12.6+（详见 [Meta SAM 3 项目](https://github.com/facebookresearch/sam3)）。本项目在本机验证过的 GPU 组合为 **PyTorch 2.14.1+cu130、torchvision 0.29.1+cu130、Python 3.13、RTX 3060 Laptop 6 GB**。新环境优先建议 Python 3.12；PyTorch 与 torchvision 必须使用彼此兼容且与显卡驱动兼容的版本。也可在安装前查看 [PyTorch 本地安装说明](https://pytorch.org/get-started/locally/)确认当前可用的 wheel。

> PyTorch 的 CUDA wheel 自带 CUDA 运行时，通常无需单独安装完整 CUDA Toolkit。仍需安装支持该 wheel 的 NVIDIA 显卡驱动。CPU 模式不需要 NVIDIA 驱动或 CUDA。

## GitHub 上传内容和本地文件

本仓库提交应用源码、依赖清单、文档和运行所需的 SAM 3 Python 源码。以下内容不会提交：

- `masked/` 目录及其所有子目录、文件（无论层级）。
- `python_embeded/`、`python_embeded.zip`、虚拟环境、缓存和编辑器配置。嵌入式 Python 运行环境体积较大，请按下方说明自行创建环境。
- `.pt`、`.pth`、`.ckpt`、`.safetensors` 等模型权重，以及 ONNX/TensorRT 模型文件。
- 上传缓存、推理结果和生成的 mask 文件。
- SAM 3 仓库中的示例、测试和开发脚本。

SAM 3 权重不包含在 GitHub 仓库中。请按 Meta SAM 3 项目的授权/访问流程获取模型权重，再放到本地 `sam3/sam3.pt`，或通过 `SAM3_CHECKPOINT` 指定其他路径。模型源代码目录默认是仓库中的 `sam3/`，可通过 `SAM3_SOURCE` 修改。

`.gitignore` 只对尚未纳入 Git 跟踪的文件生效。如果敏感文件曾经提交过，单加忽略规则不会从历史或索引中删除它；需先按数据处理要求清理 Git 跟踪记录和相关提交历史。

## 安装（Windows / NVIDIA CUDA 13.0）

以下命令在 PowerShell 中从仓库根目录执行。先安装 Python 3.12 并创建隔离环境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel "setuptools<81"
```

安装 CUDA 13.0 版 PyTorch 与匹配的 torchvision：

```powershell
python -m pip install --force-reinstall --no-deps `
  torch==2.14.1 torchvision==0.29.1 `
  --index-url https://download.pytorch.org/whl/cu130
```

安装应用及 SAM 3 推理依赖：

```powershell
python -m pip install -r requirements.txt
```

本仓库附带 SAM 3 源码。以可编辑模式登记该源码，但不让 pip 重新解析/替换已安装的 CUDA PyTorch：

```powershell
python -m pip install --no-deps -e .\sam3
```

将授权获取的检查点放到 `sam3\sam3.pt` 后，启动应用：

```powershell
python -m streamlit run app.py
```

Streamlit 启动后会在终端显示本地地址，通常为 `http://localhost:8501`。

### CUDA 安装验证

```powershell
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA wheel:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未检测到')"
```

输出应显示 CUDA 13.0 wheel、`CUDA available: True` 和 GPU 名称。若显示 `False`，先确认驱动、PyTorch wheel、Python 环境均正确，再启动应用。

## CPU 环境安装

没有 NVIDIA GPU 时，在干净的 Python 3.12 虚拟环境中安装适合 CPU 的 PyTorch/torchvision，然后执行前述 `pip install -r requirements.txt` 和 `pip install --no-deps -e .\sam3`。启动前设置 CPU：

```powershell
$env:SAM3_DEVICE = "cpu"
python -m streamlit run app.py
```

CPU 推理速度会明显慢于 GPU，且需要较多系统内存。CUDA 环境下默认使用 `cuda`；如果 CUDA 不可用，应用会报告当前 PyTorch/CUDA 状态并停止，避免误走 CPU 后只显示内存不足。需要使用 CPU 时显式设置 `SAM3_DEVICE=cpu`。

## 依赖项

`requirements.txt` 安装应用和 SAM 3 的 Python 运行依赖：

| 依赖 | 用途 |
|---|---|
| `streamlit` | Web 用户界面和本地服务。 |
| `streamlit-drawable-canvas==0.9.3` | 点提示、矩形提示和画布交互。 |
| `Pillow` | 图像读取、缩放和 PNG mask 保存。 |
| `numpy` | 图像数组、mask 合成和后处理。按 SAM 3 上游要求限制在 `<2`。 |
| `torch`、`torchvision` | SAM 3 模型推理；应按 CUDA/CPU 和平台单独安装，未放进通用 requirements。 |
| `timm` | SAM 3 图像骨干网络组件。 |
| `ftfy`、`regex` | 文本预处理和 tokenizer 依赖。 |
| `iopath`、`huggingface_hub` | 文件路径、检查点和模型资源接口。 |
| `tqdm`、`typing_extensions` | SAM 3 运行时基础依赖。 |
| `einops`、`pycocotools` | SAM 3 模型/掩码相关依赖。 |
| `psutil` | 检查系统内存和页面文件余量。 |

项目未将 Triton 设为推理必需依赖。上游某些可选后处理可能因 Triton 缺失而跳过；常规图像分割接口仍可运行。

## 配置

配置可通过环境变量覆盖，默认值见 `sam3_app/config.py`：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SAM3_DEVICE` | `cuda` | 推理设备，可设为 `cuda` 或 `cpu`。 |
| `SAM3_CHECKPOINT` | `sam3/sam3.pt` | SAM 3 检查点文件路径。 |
| `SAM3_SOURCE` | `sam3/` | SAM 3 Python 源码目录。 |
| `SAM3_CONFIDENCE` | `0.35` | 文字/框提示结果的置信度阈值。 |

PowerShell 临时设置示例：

```powershell
$env:SAM3_CHECKPOINT = "D:\models\sam3.pt"
$env:SAM3_CONFIDENCE = "0.4"
python -m streamlit run app.py
```

## 项目结构

| 路径 | 作用 |
|---|---|
| `app.py` | Streamlit 应用入口。 |
| `sam3_app/config.py` | 设备、权重、源码目录和置信度设置。 |
| `sam3_app/engine.py` | SAM 3 加载及文字、框、点提示推理适配。 |
| `sam3_app/ui.py` | 页面交互、点/框提示、结果预览和 PNG/ZIP 下载。 |
| `sam3_app/visualization.py` | 原图、分割 mask、边界框和分数叠加显示。 |
| `sam3/` | SAM 3 源码及上游许可证；模型权重需本地另行放置。 |
| `requirements.txt` | 应用与模型的通用 Python 依赖，不包含 PyTorch CUDA wheel。 |
| `.gitignore` | 忽略本地权重、虚拟环境、缓存、运行产物及 `masked/`。 |
| `.env.example` | 环境变量配置示例。 |

## 使用提示

1. 上传 PNG/JPG/JPEG/WEBP 图像。
2. 选择文字、点或框提示。
3. 点提示下先选择分组方式；手动分组时选择目标，再依次为该目标添加前景点。切换目标后继续添加其他目标的前景点。背景点可用于排除区域。
4. 框提示下选择包含框/屏蔽框标签并绘制；用“移动/调整框”修改已有框。
5. 点击“开始分割”，在结果区选择单独 mask、二值并集或目标编号 mask，并使用下载按钮保存。

### 目标编号 mask 说明

目标编号 mask 是灰度 PNG：像素值 `0` 表示背景，`1`、`2`、`3` 等表示相应的目标编号。目标重叠时，像素归给编号较小（先出现）的目标。二值并集 mask 则将所有目标前景合并为白色区域。

## 故障排查

- **找不到检查点**：确认 `sam3/sam3.pt` 存在，或检查 `SAM3_CHECKPOINT` 的路径。
- **CUDA 不可用**：确认在当前虚拟环境安装了 CUDA 版 torch/torchvision，并运行上面的验证命令。
- **内存/显存不足**：关闭占用资源的程序、减少同时运行的 GPU 应用，或使用更小图像；Windows 页面文件也应留有足够空间。
- **Streamlit 组件依赖冲突**：使用项目列出的 `streamlit-drawable-canvas==0.9.3`，不要升级到未验证的画布版本。
- **导入 SAM 3 失败**：确认 `sam3/` 目录包含上游源码，并在同一虚拟环境执行 `python -m pip install --no-deps -e .\sam3`。

