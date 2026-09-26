共享路径已于 2026-09-26 更新：数据见仓库 `data/`；初始权重见 `pretrained/`；当前生成数据位于 `data/generated/v003_yolo11m_v4e/`。旧路径记录保留用于溯源。详见 [路径重构说明](../../SHARED_LAYOUT.md)。

# v003_yolo11m_v4e：原版YOLO11m + V4E

本版本仅在推理流程使用V4E，不加入E15/E16多尺度分支、DySample或其他自定义网络模块。复用历史原版YOLO11m已训练权重，未重新训练。2026-09-22已完成三个权重的复赛V4E推理。

## 文件分区

```text
v003_yolo11m_v4e/
├─ code/
│  ├─ ultralytics/              本地Ultralytics 8.4.98源码
│  ├─ scripts/predict.py       推荐入口，检查原版结构后运行V4E
│  ├─ scripts/predict_v4e.py    当前完整V4E实现
│  ├─ src/inference/           切片与坐标辅助函数，无自定义模型模块
│  ├─ configs/                 历史训练参数、官方基础模型YAML模板
│  ├─ historical_v4e/          历史V4E代码与说明（追溯用）
│  ├─ requirements.txt
│  └─ LICENSE
├─ weights/                    epoch160.pt、epoch170.pt、epoch180.pt
├─ inference_packages/
│  ├─ chusai/epoch170_historical/  原始历史初赛V4E结果，未经改动
│  └─ fusai/epoch160、epoch170、epoch180/  本次复赛V4E结果
├─ VERSION.yaml
├─ manifest.json
└─ README.md
```

## 权重来源

三个PT均来自原项目`runs/yolo11/yolo11m_sahi_sf_continue_epoch150_to250/weights/`。这是历史原版网络训练实验，与E16的官方初始化180轮实验训练过程不同，不能直接当作完全同条件的模块消融。

默认使用epoch170，它对应原来的V4E实验基础模型。不是E16去掉分支后的权重，也不是尚未针对钢材训练的COCO预训练权重。实际模型有9类检测头；基础YAML是官方结构模板，推理直接加载PT。

## 运行命令

在工作区根目录的PowerShell执行：

```powershell
cd "<仓库根目录>"
# 仅检查三个模型结构，不训练、不推理测试集
& ..\steel-defect-yolo\.venv\Scripts\python.exe versions\v003_yolo11m_v4e\code\scripts\predict.py --epoch 160 170 180 --check-only

# 需要推理时执行；默认过滤qilie
& ..\steel-defect-yolo\.venv\Scripts\python.exe versions\v003_yolo11m_v4e\code\scripts\predict.py --epoch 170 --source "data\fusai" --dataset-name fusai
```

三个模型分别运行时使用`--epoch 160 170 180`。新输出位于`inference_packages/fusai/epoch*/`，包含V4E提交、V3E对照、标注图与缓存。V3E是V4E必需的anchor阶段，不是另一个网络模块。换测试集时设置不同的`--dataset-name`，同一目录不要并发写入。无需修改原项目或v002版本。

独立环境参考`code/requirements.txt`；当前验证环境Python3.12、PyTorch2.11.0+cu128、torchvision0.26.0+cu128。入口使用本版本的Ultralytics8.4.98源码。Windows需安装Visual C++运行库。

## V4E与qilie

1024切片、20%重叠、整图分支、自适应重切片、0.18阈值anchor、640/1280双视野弱响应确认及safe-add。模型结构保持原版。当前运行入口在完整V4E结束后过滤qilie，其余类别输出不变；保留9类模型，避免改变已训练检测头。

## 历史结果边界

`chusai/epoch170_historical`保存过去生成的V4E初赛JSON/ZIP，不适用于复赛图片。历史说明记载2825框、Recall=0.8380、Precision=0.2949、F1=0.4362、mAP@0.5=0.4943；属于历史初赛记录，未在此次新建版本时重新验证平台成绩。

历史脚本依赖当时的首轮缓存和V3E anchor，因此作为追溯材料保存。当前完整V4E脚本来自后来按伪代码实现的版本，聚类等细节已固定，但尚未证明它与历史脚本逐框完全等价。历史提交保持原样（可能包含qilie），新复赛提交统一过滤qilie。

## 2026-09-22复赛推理结果

使用本版本原版网络，测试集为`复赛测试集/复赛发布`，每组788张图；最终提交已过滤qilie，平台成绩尚未评测。

| Epoch | V3E原框 | V4E新增 | V4E最终框数 | 提交包 |
|---|---:|---:|---:|---|
|160|2169|248|2417|inference_packages/fusai/epoch160/v4e_submission.zip|
|170|2188|250|2438|inference_packages/fusai/epoch170/v4e_submission.zip|
|180|2208|249|2457|inference_packages/fusai/epoch180/v4e_submission.zip|

已核对每组788张图的记录、缓存与标注图，ZIP内部JSON与磁盘文件一致，qilie为0。目录中同时保留V3E中间结果以便审计；正式使用上表V4E包。未进行模型训练。文件清单包含提交文件与汇总，逐图缓存及标注图片不逐个列入manifest。

## 自动批量推理复赛测试集

新增本版本独立入口 `START_INFERENCE.cmd` / `INFER_FUSAI.py`，自动读取本版本weights中的所有PT，依次运行V4E，输出至本版本inference_packages/fusai_auto。使用方法与断点缓存说明见 [INFERENCE_README.md](INFERENCE_README.md)。本次仅添加代码，未执行复赛批量推理。
