共享路径已于 2026-09-26 更新：数据见仓库 `data/`；初始权重见 `pretrained/`；当前生成数据位于 `data/generated/v004_e16_v4e_split/`。旧路径记录保留用于溯源。详见 [路径重构说明](../../SHARED_LAYOUT.md)。

旧训练合同未改写；重构前权重仍可用于推理，新训练请创建新版本。下方历史恢复命令不适用于重构前合同。

# v004_e16_v4e_split：YOLO11m + E16输出残差分支 + V4E

本版本用于现有2560张训练图 / 640张本地验证图的独立对比。从官方yolo11m.pt重新初始化，不使用已见过全部3200张图的旧项目权重。代码已生成，尚未执行正式数据切片、训练或640张验证推理。

## 文件结构

```text
v004_e16_v4e_split/
├─ code/
│  ├─ ultralytics/              本地Ultralytics 8.4.98源码
│  ├─ scripts/                  数据准备、训练、V4E推理、评测、归档
│  ├─ configs/experiment.yaml   本版本数据、训练与评测设置
│  ├─ configs/yolo11m.yaml      9类YOLO11m基础结构
│  ├─ pretrained/yolo11m.pt     历史副本；当前使用仓库 pretrained/yolo11m.pt
│  ├─ src/                      切片工具及E16结构
│  ├─ tests/                    合成数据与检查点测试
│  ├─ data/                     历史缓存保留；当前生成到仓库 data/generated/<版本>/
│  ├─ training_records/train/   运行训练后生成：参数、损失、验证指标、恢复记录
│  ├─ artifacts/                检查报告、运行时缓存
│  ├─ requirements.txt
│  └─ LICENSE
├─ weights/                    正式训练后生成best.pt、last.pt、epoch10.pt…epoch180.pt
├─ inference_packages/         local_val/epochN/ 或 自定义数据集名/epochN/
├─ README.md
├─ VERSION.yaml
└─ manifest.json
```

训练权重不存入code目录。训练记录不会自动塞进inference_packages。V4E是推理流程，两版都使用同一份v4e_core.py。

## 网络和对比条件

E16只在model.16的完整C3k2输出后添加256→128→并行3/5/7深度卷积→拼接→256的残差分支，末端投影零初始化，增加141696参数。最近邻上采样与三尺度检测头保持不变。基础YAML单独加载不含E16，必须经本版本train.py构建。

两版本唯一实验差异是architecture字段及E16模块；训练参数、数据准备策略、官方初始化文件和V4E代码相同。参数默认180轮、1024输入、batch4、nbs64、AdamW lr0=0.001、热身3轮、余弦衰减、mosaic=0.8、最后20轮关闭Mosaic。配置见code/configs/experiment.yaml，首轮不同时叠加新的采样、强增强或损失改动。

## 数据范围与防泄漏

使用 ../../data/splits/d001_train_test_4to1，保持其split_manifest.csv固定划分。目录名test的640张在本项目中作为开发验证集，data.yaml中的val/test别名不是两份独立数据。

- train=2560张，4707个有效框；val=640张，1178个有效框。
- 每次入口核对标签SHA256、类别顺序、图片清单、来源组、原图SHA256组不跨集合；--verify-images还能重新验证全部源图片字节。
- 仅2560张训练图产生1024/20%重叠切片，并保留一份训练整图。空切片保留概率0.10沿用原基线，**不是最终背景比例10%**。
- 验证只生成640张完整图的独立加载清单，不生成训练增强、不加入训练切片。
- 全图与标签优先用本机硬链接节省空间，跨磁盘则复制；不要手工编辑生成缓存中的硬链接文件，否则也会影响数据副本。切片JPEG是独立文件。
- 每张原图的背景抽样使用固定局部随机种子，中断后重跑prepare_data可复用完整记录；配置变化会拒绝混用旧缓存。
- 验证默认全部9类，包含qilie；不会沿用历史复赛的隐式过滤。自定义外部推理若确需过滤，显式传--exclude-class qilie。

## 本机运行

在 yolo11-source-workspace 目录中使用PowerShell。以下命令逐条执行，两版应顺序训练，避免争抢同一块8GB显卡。

```powershell
cd "<仓库根目录>"

# 1. 检查划分与官方初始化模型，合成输入前向，不训练、不推理真实验证图
& ..\steel-defect-yolo\.venv\Scripts\python.exe versions\v004_e16_v4e_split\code\scripts\train.py --check

# 2. 生成本版训练切片、整图与验证加载清单；首次需要时间和磁盘空间
& ..\steel-defect-yolo\.venv\Scripts\python.exe versions\v004_e16_v4e_split\code\scripts\prepare_data.py --verify-images

# 3. 正式训练；只有明确运行这条命令才开始训练
& ..\steel-defect-yolo\.venv\Scripts\python.exe versions\v004_e16_v4e_split\code\scripts\train.py

# 中断后恢复同一训练，已完成180轮的任务不能再作为中断恢复
& ..\steel-defect-yolo\.venv\Scripts\python.exe versions\v004_e16_v4e_split\code\scripts\train.py --resume

# 4. 使用best.pt运行640张完整验证图的V4E，并自动统计本地指标
& ..\steel-defect-yolo\.venv\Scripts\python.exe versions\v004_e16_v4e_split\code\scripts\evaluate.py

# 或比较指定轮数；epoch编号按人类习惯从1开始
& ..\steel-defect-yolo\.venv\Scripts\python.exe versions\v004_e16_v4e_split\code\scripts\evaluate.py --epoch 170
```

只检查数据而不切片：`prepare_data.py --check`。模型检查报告在code/artifacts/model_check.json；--check通过不等于完成整轮训练或获得准确率。

## 本地验证与模型选择

训练期间val=true，每轮做完整原图缩放到1024的Ultralytics验证；best.pt按该验证fitness选择，**不是按V4E召回率选出的最优权重**。需要用evaluate.py对保存的epoch进行相同V4E评测，才能选择实际部署权重。本入口不自动每10轮运行V4E，避免额外推理任务和GPU争用。

evaluate.py包含全部640张图片，零检测图仍记录，背景误检计入FP。基于同类、IoU≥0.50、一对一匹配，输出Precision/Recall/F1、101点AP50、TP/FP/FN、每类指标、短边/长边分组和两类文件名来源代理的指标。AP仅基于V4E导出的已阈值筛选框，不等于原检测器低阈值完整PR曲线AP，也不是官方计分公式。

已存在结果可只计算指标而不重复推理：

```powershell
& ..\steel-defect-yolo\.venv\Scripts\python.exe versions\v004_e16_v4e_split\code\scripts\evaluate.py --predictions versions\v004_e16_v4e_split\inference_packages\local_val\epoch170\v4e_predictions.json
```

qilie验证只有6张、14框，需同时看TP/FN计数。参数调整使用这640张后，其角色是开发验证，不能再称从未参与选模的最终测试集。

## V4E推理与输出

固定沿用归档V4E核心：1024切片/20%重叠+整图；至多4个自适应1024重切片；0.18阈值形成V3E anchor；0.05–0.18弱候选聚类后选至多2个区域；640/1280两视野任一确认，阈值0.10；按类NMS=0.50，保留anchor再safe-add。单次模型NMS=0.70、conf=0.05、max_det=1000。核心常量在scripts/v4e_core.py，与当前归档完整实现一致，未宣称与更早历史缓存实现逐框等价。

默认predict.py只跑本地验证集。外部图片需显式给source和dataset-name，不会自动指向初赛或复赛：

```powershell
& ..\steel-defect-yolo\.venv\Scripts\python.exe versions\v004_e16_v4e_split\code\scripts\predict.py --epoch 170 --source "data\fusai" --dataset-name fusai --exclude-class qilie --visualize
```

输出位于inference_packages/<数据集名>/epochN/：v4e_predictions.json、v4e_submission.json/zip、V3E对照、逐图缓存、进度和摘要；evaluate额外生成metrics.json与evaluation_report.md。ZIP中使用submission.json。--visualize才生成标注图。bbox导出为原图整数坐标，舍弃取整后退化的零面积框。

推理必须加载本版新划分训练产生且带有来源指纹的权重；不接受历史全量模型来伪装独立验证。缓存校验模型、源图片字节、代码、过滤设置和设备；不匹配时用新的dataset-name。默认保留9类；想改变过滤口径时不能覆盖原结果。

## 恢复、归档及限制

last.pt和每10轮权重保留FP32 live模型、EMA、优化器、AMP scaler和调度器；恢复核对数据/源码/配置指纹和优化器参数顺序。按完整epoch恢复，多进程增强随机序列不承诺逐位复现。NaN/OOM时停止，不自动换batch或隐式重建优化器。完成训练不调用strip_optimizer，保留可恢复状态。

源码或训练超参数变化应另建版本；同版重跑需--resume，不覆盖历史实验。脚本在训练退出、推理完成和评测完成时更新manifest及下方真实状态；运行中不使用档案校验作为一致性快照。manifest排除生成数据、逐图缓存、可视化及运行时artifacts，范围明示在文件中。

CPU可在训练配置设device: cpu，推理传--device cpu；本机训练默认单GPU。依赖复用原项目环境，独立电脑需要匹配PyTorch/torchvision和Visual C++运行库。源码包含AGPL-3.0许可证。

## 当前运行状态

<!-- RUN_STATUS_START -->
已完成训练轮数：0。尚未训练，无新训练权重。

本地V4E尚未评测；无平台成绩。
<!-- RUN_STATUS_END -->

## 自动批量推理复赛测试集

新增本版本独立入口 `START_INFERENCE.cmd` / `INFER_FUSAI.py`，自动读取本版本weights中的所有PT，依次运行V4E，输出至本版本inference_packages/fusai_auto。使用方法与断点缓存说明见 [INFERENCE_README.md](INFERENCE_README.md)。本次仅添加代码，未执行复赛批量推理。
