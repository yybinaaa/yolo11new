共享路径已于 2026-09-26 更新：数据见仓库 `data/`；初始权重见 `pretrained/`；当前生成数据位于 `data/generated/v002_e16_v4e/`。旧路径记录保留用于溯源。详见 [路径重构说明](../../SHARED_LAYOUT.md)。

# v002_e16_v4e — E16 模型与 V4E 推理归档

这是已经完成训练的 E16 版本，包含第160、170、180轮完整权重和可直接运行的V4E推理代码。代码、PT文件、推理包分别存放在`code/`、`weights/`、`inference_packages/`。模型采用YOLO11m，在第16层C3k2的**完整输出后**增加多尺度深度卷积残差分支。V4E是推理策略名称，不是模型结构名称。

## 目录

```text
v002_e16_v4e/
├─ code/                         代码、配置、依赖说明
│  ├─ ultralytics/               Ultralytics 8.4.98 本地源码快照
│  ├─ src/models/                E16 自定义结构及共享分支定义
│  ├─ src/inference/             切片与坐标辅助函数
│  ├─ scripts/predict.py         推荐入口，按epoch选择模型
│  ├─ configs/                   模型基础图配置
│  ├─ training_archive/          训练代码、配置和原记录（追溯用）
│  ├─ requirements.txt
│  └─ LICENSE
├─ weights/epoch160.pt
├─ weights/epoch170.pt
├─ weights/epoch180.pt
├─ inference_packages/epoch160/  已过滤qilie的复赛JSON/ZIP及校验报告
├─ inference_packages/epoch170/
├─ inference_packages/epoch180/
├─ VERSION.yaml
├─ manifest.json                 文件SHA256，用于归档校验
└─ outputs/                      新推理结果，按epoch分目录
```

`src.models.multiscale_dw_output.MultiScaleOutputC3k2`等导入路径必须保留，权重反序列化依赖这些类。

## 当前电脑直接运行

在PowerShell执行，复用已有环境：

```powershell
cd "<仓库根目录>"
& ..\steel-defect-yolo\.venv\Scripts\python.exe versions\v002_e16_v4e\code\scripts\predict.py --epoch 160 170 180 --source "data\fusai"
```

只运行180轮：

```powershell
& ..\steel-defect-yolo\.venv\Scripts\python.exe versions\v002_e16_v4e\code\scripts\predict.py --epoch 180 --source "data\fusai"
```

新结果默认保存于本版本`outputs/epoch160`、`outputs/epoch170`、`outputs/epoch180`。每个目录有`v4e_submission.zip`、`v4e_submission.json`、V3E对照包、标注图片、逐图缓存和汇总。换数据集请使用新的`--output-root`目录。重复执行同一任务会检查文件清单、权重和代码哈希，然后复用已完成图片的缓存。不要同时运行两个写入同一目录的实例。

本版本不通过工作区根目录的通用`train.py`或`evaluate.py`启动；根目录当前默认版本仍为`v001_baseline`。本版本入口会显式使用本版本的Ultralytics源码和自定义结构，不依赖原项目源码。原项目仅提供现有Python环境，数据路径由`--source`指定。

## 其他电脑环境

验证环境为Python3.12、PyTorch2.11.0+cu128、torchvision0.26.0+cu128及Ultralytics8.4.98。创建环境后先安装匹配硬件的PyTorch/torchvision，再安装`code/requirements.txt`中其余依赖。Ultralytics源码已经随版本提供。Windows需要Microsoft Visual C++运行库；附带的runtime_bootstrap是当前电脑兼容措施，独立电脑应安装正常运行库。

在本版本目录运行：

```text
python code/scripts/predict.py --epoch 180 --source /path/to/images --device 0
```

CPU可使用`--device cpu`，速度会明显降低。

## 模型与推理

E16结构：`y=C3k2(x)`，`output=y+MultiScaleDWBranch(y)`。分支采用1×1降维、并行3×3/5×5/7×7深度卷积、拼接及1×1恢复通道，末端投影零初始化；第16层为256通道，新增141696参数。第11、14层保持最近邻上采样。

V4E沿用已测试流程：1024切片、20%重叠及整图推理；最多4个1024自适应重切片；0.18阈值生成V3E；弱响应区域最多2个，以640和1280双视野复检；保持V3E原框，只追加确认的新框。跨视野NMS阈值0.50，模型单次预测置信度0.05、NMS阈值0.70、max_det=1000。

最终导出和可视化统一过滤`qilie`，其余8类保留：gunyin、huashang、jiaza、jieba、mamianmakeng、yanghuatiepi、yiwuyaru、zonglie。为复现之前已提交的结果，过滤发生在完整V4E流程之后；模型仍为9类模型，原始候选缓存可能包含qilie，提交JSON和标注结果不包含它。

## 已有复赛提交包

复赛共788张图。以下为已完成结果，过滤只移除qilie，不改变其他框或分数：

| Epoch | V4E提交框数 | 目录 |
|---|---:|---|
|160|3333|inference_packages/epoch160/|
|170|3314|inference_packages/epoch170/|
|180|3251|inference_packages/epoch180/|

各目录内提交文件名包含epoch和`no_qilie`。V3E对照包也已保存。框数不是精度指标；本版本未附隐藏测试集真值或平台评分。

## 训练记录与复现范围

训练从官方yolo11m.pt初始化，共180轮，AdamW、lr0=0.001、3轮热身、余弦衰减、1024输入、batch4，最后10轮关闭Mosaic；保存160/170/180轮。`code/training_archive/`仅供追溯，保留原始路径与训练指纹，不能在此目录直接启动归档训练脚本。数据集与官方初始权重未随版本复制，独立复训需在原项目使用`train_e16_pretrained.py`并配置数据。三个已训权重保留完整检查点内容，但直接在迁移目录恢复原任务会触发原路径/指纹校验，不能当作可直接迁移的续训入口。

本次打包不启动训练；仅对三个权重进行单图推理与提交过滤复现检查。许可证见`code/LICENSE`（AGPL-3.0）。

三个权重的SHA256均与原训练输出一致；三个权重在包含气裂候选的同一张复赛图上，过滤后的预测与原提交逐项完全一致。227个Ultralytics Python源码文件与原训练环境安装版本逐文件一致。后续可运行`python code/scripts/verify_release.py`核对本归档清单。

## 自动批量推理复赛测试集

新增本版本独立入口 `START_INFERENCE.cmd` / `INFER_FUSAI.py`，自动读取本版本weights中的所有PT，依次运行V4E，输出至本版本inference_packages/fusai_auto。使用方法与断点缓存说明见 [INFERENCE_README.md](INFERENCE_README.md)。本次仅添加代码，未执行复赛批量推理。
