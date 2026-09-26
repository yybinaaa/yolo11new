# 共享目录已整理（v001～v005）

数据统一位于 `data/`，官方初始权重位于 `pretrained/`。运行、创建新版本与历史记录说明见 [SHARED_LAYOUT.md](SHARED_LAYOUT.md)。环境与 v006 保持原样。

# YOLO11 源码迭代工作区

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

这个目录用于直接修改 YOLO11 源码，并让每次模型迭代都能追溯到独立源码快照。它不会创建或安装新的虚拟环境，运行时复用当前已有的 Python、PyTorch 和 CUDA 环境。

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

## 当前环境运行

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

不要覆盖 `v001_baseline`。例如创建 DySample 版本：

```powershell
& ..\steel-defect-yolo\.venv\Scripts\python.exe scripts\new_version.py v002_dysample --description "在 YOLO11m Neck 中加入 DySample"
```

然后只修改：

```text
versions/v002_dysample/ultralytics/...
versions/v002_dysample/models/yolo11m.yaml
```

所有超参数差异继续写入唯一的 `configs/experiment.yaml`：在 `versions.v002_dysample.overrides` 下填写，并将 `active_version` 改成目标版本。也可通过 `--version v002_dysample` 临时选择版本。

`model.scale` 可选 `n/s/m/l/x`，它会自动选择对应的 `models/yolo11{scale}.yaml`。如果更换尺度，也要把 `paths.pretrained_weights` 改为同尺度权重；修改了网络结构时，将 `model.build_from_yaml` 设为 `true`，这样会先按修改后的 YAML 构图，再迁移能匹配的预训练参数。

训练/评测启动时会把解析后的完整配置、权重路径和源码 SHA256 写进 `artifacts/launches/<版本>/`，便于复现实验和确认训练时究竟用了哪份源码。

## 修改源码时的关键位置

- 网络基础模块：`ultralytics/nn/modules/`
- 模型解析与构建：`ultralytics/nn/tasks.py`
- 检测训练器：`ultralytics/models/yolo/detect/train.py`
- 检测验证器：`ultralytics/models/yolo/detect/val.py`
- YOLO11 结构：`models/yolo11m.yaml`

本地源码继承上游 Ultralytics 的 AGPL-3.0 许可证，许可证副本位于每个版本目录中。

