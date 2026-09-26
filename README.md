# YOLO11 源码迭代工作区

v001～v005 的共享数据统一位于 `data/`，官方初始权重位于 `pretrained/`。运行、创建新版本与历史记录说明见 [SHARED_LAYOUT.md](SHARED_LAYOUT.md)。v006 尚未迁移共享路径。

## 本机运行环境

以下为项目记录的本机环境（2026-09-21），在 README 正文列出，便于直接查看和复现：

| 项目 | 版本或配置 |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA 运行时 | 12.8 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Ultralytics 本地源码 | 8.4.98 |
| NumPy | 2.5.1 |
| OpenCV | 5.0.0（对应安装包 opencv-python 5.0.0.93） |
| PyYAML | 6.0.3 |
| Pandas | 3.0.3 |
| Matplotlib | 3.11.0 |

这是已有环境记录；显卡型号不是必须相同。Pandas 属于原环境中的包，v005 的直接依赖清单未将其列为必装项。创建虚拟环境和安装固定版本的步骤如下。

## 从零配置环境（Windows，优先使用 v005）

以下命令在 **Windows PowerShell** 中执行。除克隆仓库外，其余命令均在仓库根目录执行。已有可用环境的用户不需要重装；本节供新电脑安装使用。

### 1. 准备软件

- 安装 **64 位 Python 3.12**，并启用 Python Launcher（`py` 命令）。本项目已有运行环境为 Python **3.12.14**；其他 3.12 补丁版本未逐一测试。
- 安装 Git 和 [Git LFS](https://git-lfs.com/)。LFS 用于下载本仓库的 `epoch170.pt`。
- GPU 路径需要 NVIDIA 显卡及支持 CUDA 12.8 的驱动，可先运行 `nvidia-smi` 检查驱动是否正常。PyTorch 选择下述 CUDA wheel；不需要为了本项目额外编译 CUDA。
- Windows 如果出现 `msvcp140.dll`、`VCRUNTIME140.dll` 缺失或相关 DLL 加载失败，先安装或修复微软的 [Visual C++ x64 运行库](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)。不需要安装 Codex 提供 DLL。

GPU 安装版本依据 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/#v2110)。本机使用 RTX 4070 Laptop GPU；未对其他硬件作完整训练验证。

### 2. 下载代码和已发布权重

```powershell
git lfs install
git clone https://github.com/yybinaaa/yolo11new.git
cd yolo11new
git lfs pull --include="versions/v005_yolo11m_v4e_split/weights/epoch170.pt" --exclude=""
```

如果已经克隆仓库，进入已有目录并执行 `git lfs pull` 即可。不要把 GitHub 页面里的约百字节 LFS 指针当作 PT 模型；完整 `epoch170.pt` 大小为 **322413021 字节**。

### 3. 建立虚拟环境

```powershell
py -3.12 --version
py -3.12 -m venv .venv
```

后续命令直接指定 `.venv` 中的 Python，无需激活环境，也不依赖旧电脑的 `steel-defect-yolo/.venv`。
如果希望激活，可执行：

```powershell
.\.venv\Scripts\Activate.ps1
```

若 PowerShell 阻止激活脚本，直接继续使用下面的完整解释器路径即可，无需更改系统执行策略。

### 4. 按固定版本安装依赖

**NVIDIA GPU 环境（与现有环境匹配）**，逐条执行：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r versions/v005_yolo11m_v4e_split/code/requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

第二条指定 CUDA 12.8 构建；第三条一并安装版本文件中固定的其他依赖，已安装的匹配 PyTorch 构建会保留。任一步失败先处理该错误，不要直接跳到训练。

核心版本如下，完整直接依赖以 [v005 requirements.txt](versions/v005_yolo11m_v4e_split/code/requirements.txt) 为准：

| 包 | 固定版本 |
|---|---|
| torch | 2.11.0（GPU 构建为 `2.11.0+cu128`） |
| torchvision | 0.26.0（GPU 构建为 `0.26.0+cu128`） |
| numpy | 2.5.1 |
| opencv-python | 5.0.0.93 |
| Pillow | 12.3.0 |
| PyYAML | 6.0.3 |
| scipy | 1.18.0 |
| matplotlib | 3.11.0 |
| psutil | 7.2.2 |
| polars | 1.42.1 |
| ultralytics-thop | 2.0.20 |
| requests | 2.34.2 |
| tqdm | 4.68.4 |
| Ultralytics 源码 | 仓库内随版本归档的 8.4.98 |

**不要额外执行 `pip install ultralytics` 或升级它。** 各版本入口会加载该版本自带的源码，以保证模型结构和检查点兼容。上表固定直接依赖，传递依赖仍由 pip 解析；`environment-current.txt` 是原机器的环境记录，不是安装清单。

仅 CPU 使用时，把上面第二条的索引地址替换为 `https://download.pytorch.org/whl/cpu`，其余步骤相同；推理参数使用 `--device cpu`。CPU 路径适合检查或小规模推理，未进行完整训练复现。

### 5. 检查环境

```powershell
.\.venv\Scripts\python.exe -c "import sys, torch, torchvision, cv2, numpy, yaml; print('Python:', sys.version); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

GPU 安装应显示 `torch: 2.11.0+cu128`、`CUDA runtime: 12.8` 和 `CUDA available: True`。若为 False，先检查 NVIDIA 驱动和所安装的 PyTorch 构建。

### 6. 用 v005 第 170 轮权重检查和推理

准备一个放有待检测图片的目录，把示例中的 `D:\steel-images` 换成你的实际路径。这个入口不要求完整训练集。

```powershell
# 只加载检查点、检查输入与类别，不预测图片
.\.venv\Scripts\python.exe versions/v005_yolo11m_v4e_split/INFER_FUSAI.py --weights epoch170.pt --source "D:\steel-images" --device 0 --check

# 确认检查通过后，下面这条才执行真实图片推理
.\.venv\Scripts\python.exe versions/v005_yolo11m_v4e_split/INFER_FUSAI.py --weights epoch170.pt --source "D:\steel-images" --device 0
```

输出保存在该版本的 `inference_packages/fusai_auto/epoch170/` 下，按运行配置区分目录。复赛入口默认排除 `qilie`；需要保留全部 9 类时加 `--keep-qilie`，需要标注图时加 `--visualize`。

原 `START_*.ps1/.cmd` 仍默认寻找旧项目的环境。使用新 `.venv` 时优先使用上面的 Python 命令；也可在仓库根目录显式指定解释器：

```powershell
.\versions\v005_yolo11m_v4e_split\START_INFERENCE.ps1 -PythonExe "$PWD\.venv\Scripts\python.exe" -Source "D:\steel-images" -Weights epoch170.pt -CheckOnly
```

### 7. 训练前还需要准备什么

当前 GitHub **仅发布 v005 的 epoch170.pt，未发布数据集和官方初始权重**。安装环境不等于具备训练所需全部文件：

- 固定划分的图片、标注与清单放到 `data/splits/d001_train_test_4to1/`；保持原有 2560/640 划分。
- 官方初始化权重放到 `pretrained/yolo11m.pt`，不要用已训练的 9 类 epoch170.pt 冒充初始化权重。
- 新训练先创建独立版本，按 [共享目录说明](SHARED_LAYOUT.md) 和新版本 README 操作。历史 v005 已训练完成，不覆盖其已有结果。

准备好训练数据后，可先只校验文件：

```powershell
.\.venv\Scripts\python.exe versions/v005_yolo11m_v4e_split/code/scripts/prepare_data.py --check --verify-images
```

本节安装命令已对照本机实际安装版本和 PyTorch 官方索引核对；尚未在全新电脑上完整重装验证。以下保留各版本介绍及旧基线入口说明。

## v006：E16＋P3/P4空间门控＋V4E

[v006_e16_p3p4_gate_v4e](versions/v006_e16_p3p4_gate_v4e/README.md)基于v002独立建立，保留E16多尺度残差分支，新增32通道P3/P4空间门控和`2×Sigmoid`恒等初始化。按用户要求复用全部3200张训练图对应的原混合数据，不再划分本地验证集，不使用v004/v005划分。代码、初始化权重、训练PT及按评测集分组的推理结果分别归档。训练入口默认只检查，显式`--train`才训练；建版未启动正式训练或真实评测集推理。请使用版本内`code/scripts/train.py`与`predict.py`，不要使用根目录旧入口。

## 按新划分重新训练的两版代码

- [v004_e16_v4e_split](versions/v004_e16_v4e_split/README.md)：E16 + V4E。
- [v005_yolo11m_v4e_split](versions/v005_yolo11m_v4e_split/README.md)：原始YOLO11m + V4E。

两版固定使用d001的2560/640划分，从官方预训练重新初始化；训练参数与V4E流程一致，只比较E16结构。默认本地验证保留全部9类。每版提供prepare_data.py、train.py、predict.py、evaluate.py以及恢复训练和归档校验入口；命令见对应README。已完成构图、数据隔离和合成检查点测试，尚未生成正式切片、未训练、未在640张上推理评测。

请使用各版本code/scripts中的入口；工作区根目录旧train.py及默认experiment.yaml不驱动这两版。历史v002/v003权重和结果继续保留。

## 本地 4:1 训练/测试数据集

[d001_train_test_4to1](data/splits/d001_train_test_4to1/README.md) 将原始 3,200 张带 XML 标注图片按来源组及完全重复文件合组，划分为 2,560 张训练图和 640 张测试图，两侧均覆盖 9 类。包含独立图片副本、原始 XML、YOLO 标签、固定划分清单、分类统计和 SHA256 校验；原始数据保留不变。

新数据入口为 `data/splits/d001_train_test_4to1/data.yaml`，当前默认训练配置尚未切换。后续新实验应独立建版本，只从训练部分生成裁块与增强数据，不能混用旧全量缓存。历史模型已经使用过全部原图，不能将这份划分作为旧模型的独立测试。该数据集只有训练/测试两份；配置中 `val` 与 `test` 指向同一测试目录，如果用于调参或选 epoch，需将其视为验证集。此次仅完成数据划分，未训练或评测。

## 原版网络 + V4E

[v003_yolo11m_v4e](versions/v003_yolo11m_v4e/README.md)保留原版YOLO11m结构，仅使用V4E推理，包含历史原版160/170/180轮权重。按`code/`、`weights/`、`inference_packages/`分别归档，不包含E15/E16模块。已完成三个权重的788张复赛图推理，结果位于该版本`inference_packages/fusai/epoch*/`，最终提交过滤qilie。

## 已训练的 E16 + V4E 版本

新增独立版本 [v002_e16_v4e](versions/v002_e16_v4e/README.md)，内含本地Ultralytics源码、自定义E16模块、第160/170/180轮权重、V4E推理入口，以及过滤`qilie`后的复赛提交包。当前默认训练版本仍是`v001_baseline`。

在本目录执行三个权重的复赛推理：

```powershell
& ..\steel-defect-yolo\.venv\Scripts\python.exe versions\v002_e16_v4e\code\scripts\predict.py --epoch 160 170 180 --source "data\fusai"
```

详见版本README；该归档版本使用自己的推理入口，训练记录仅用于追溯。

这个目录用于直接修改 YOLO11 源码，并让每次模型迭代都能追溯到独立源码快照。训练脚本不会自动创建或安装环境；可使用上文新建的 `.venv`，也可以继续复用已有环境。

## 目录结构

```text
yolo11-source-workspace/
├─ configs/experiment.yaml        # 所有模型、训练、评测超参数
├─ scripts/train.py               # 训练入口
├─ scripts/evaluate.py            # 评测入口
├─ scripts/check_environment.py   # 环境与源码检查
├─ scripts/new_version.py         # 从旧版本复制出新源码版本
├─ versions/
│  └─ v001_baseline/
│     ├─ ultralytics/             # YOLO11/Ultralytics 8.4.98 源码
│     ├─ models/                  # n/s/m/l/x 五种 YOLO11 模型 YAML
│     └─ VERSION.yaml
├─ runs/                          # 训练和评测输出
└─ artifacts/                     # 每次启动的完整配置和源码哈希
```

## 旧基线入口（v001，已有完整本地数据时使用）

如果终端已经进入原项目环境，直接运行：

```powershell
cd C:\Users\16125\Desktop\钢材AI\yolo11-source-workspace
python scripts/check_environment.py
python scripts/train.py --dry-run
```

如果终端没有激活环境，可直接复用原项目已有解释器，不会创建新环境：

```powershell
& ..\steel-defect-yolo\.venv\Scripts\python.exe scripts\check_environment.py
& ..\steel-defect-yolo\.venv\Scripts\python.exe scripts\train.py --dry-run
```

确认无误后，去掉 `--dry-run` 开始训练：

```powershell
& ..\steel-defect-yolo\.venv\Scripts\python.exe scripts\train.py
```

评测默认自动寻找当前版本最近生成的 `best.pt`，也可以明确指定：

```powershell
& ..\steel-defect-yolo\.venv\Scripts\python.exe scripts\evaluate.py --weights runs\v001_baseline\train\weights\best.pt
```

当前数据配置的 `val` 实际指向训练集，评测脚本会明确警告。这套数据只能验证流程，不能作为独立泛化性能结论；获得正式验证集后，只需修改 `configs/experiment.yaml` 的 `paths.data`。

## 创建下一迭代版本

保留已有版本。从 v005 创建新实验的示例（仅创建代码，不训练）：

```powershell
.\.venv\Scripts\python.exe scripts\new_version.py v007_yolo11m_example --from-version v005_yolo11m_v4e_split --description "本次模型改进说明"
```

然后只修改：

```text
versions/v007_yolo11m_example/code/ultralytics/...
versions/v007_yolo11m_example/code/configs/yolo11m.yaml
versions/v007_yolo11m_example/code/configs/experiment.yaml
```

v004/v005 派生版本使用版本内 `code/configs/experiment.yaml`，共享数据和初始权重，独立保存生成缓存及结果。创建后按新版本 README 操作，不需要修改根目录的 `active_version`。

只有从 v001 派生的旧基线结构使用根目录 `configs/experiment.yaml` 的版本 overrides、`model.scale` 和 `--version` 入口。不要与 v005 的版本内训练入口混用。

v005 派生版本的训练记录保存在该版本 `code/training_records/`；旧基线入口的启动记录保存在仓库 `artifacts/launches/<版本>/`。

## 修改源码时的关键位置

- 网络基础模块：`ultralytics/nn/modules/`
- 模型解析与构建：`ultralytics/nn/tasks.py`
- 检测训练器：`ultralytics/models/yolo/detect/train.py`
- 检测验证器：`ultralytics/models/yolo/detect/val.py`
- YOLO11 结构：`models/yolo11m.yaml`

本地源码继承上游 Ultralytics 的 AGPL-3.0 许可证，许可证副本位于每个版本目录中。
