# v006：E16＋P3/P4空间门控＋V4E

基于v002源码独立建立；原版本保持不变。保留第16层完整C3k2后的E16多尺度深度卷积残差分支，新增32通道压缩的P3/P4空间门控。此次建版不进行真实数据训练或评测集推理，不复制旧版本结果冒充新模型成绩。

## 文件布局

```text
v006_e16_p3p4_gate_v4e/
├─ code/
│  ├─ ultralytics/            v002的Ultralytics 8.4.98源码
│  ├─ src/models/             原E16模块＋p3p4_spatial_gate.py
│  ├─ src/inference/          原V4E辅助工具
│  ├─ scripts/                训练、推理、检查与归档入口
│  ├─ configs/                模型基础图、训练与推理配置
│  ├─ training_records/       新训练记录，未复制v002训练成绩
│  ├─ tests/                  合成输入结构与梯度验证
│  ├─ artifacts/              数据、构图、验证报告
│  ├─ requirements.txt
│  └─ LICENSE
├─ weights/
│  ├─ initialization/yolo11m.pt  官方初始化PT，非v006训练成果
│  └─ last.pt、epoch160/170/180.pt  正式训练后生成
├─ inference_packages/       <评测集名称>/<权重名称>/，推理后生成
├─ README.md
├─ VERSION.yaml
└─ manifest.json
```

## 网络结构

`X=C3k2(layer15)`；`R=E16(X)`保持原1×1降维、并行3×3/5×5/7×7深度卷积、拼接及1×1投影。

P3使用X（256通道），P4使用第13层输出（512通道）。各经1×1卷积压缩至32通道、SiLU；P4双线性插值对齐P3，再拼接为64通道，3×3卷积至32通道、SiLU、1×1至1通道。`G=2*sigmoid(logits)`，`Y=X+G*R`。

门控末层权重和偏置置零，初始G=1，精确保留同权重E16计算。新增43,041参数。原E16末端零投影初始化规则不变；若从已训练E16初始化，则保留其完整分支权重，不再次置零。门控只控制残差，不直接抹除X。门控权重不是缺陷概率。

第16层from改为`[-1,13]`，缓存第13层输出。原第11/14层最近邻上采样、三尺度Detect `[16,19,22]`及stride 8/16/32保持不变；不增加P2检测头。第16层变化也会通过PAN传到后续层。新增类必须保留导入路径`src.models.p3p4_spatial_gate`以加载PT。

## 初始化与训练数据

默认沿用v002训练方式：从官方yolo11m.pt初始化，180轮，1024输入，batch4，AdamW lr0=0.001，3轮热身，余弦衰减，最后10轮关闭Mosaic。该PT随版本放入weights/initialization，SHA256在manifest和检查报告中。

复用原项目`steel-defect-yolo/data_cache/sahi_sf_mixed_1024_overlap20/data.yaml`，包含全部3200张原图对应的12429个切片＋3200份整图，共15629条。不使用v004/v005的2560/640划分，不创建新的本地划分。检查入口核对训练清单、原图来源数、标签内容与SHA256、图片存在性/大小/修改时间；不宣称重新逐字节审计所有图像。

配置中val=false。继承的数据YAML含val=train的框架占位字段，训练器禁止实际验证，最后一轮也不运行验证；不生成或宣称本地验证精度，不选择best.pt。你提供的外部评测集仅用于明确启动的推理，不进入本版本训练数据。

若要从已训练E16微调，训练前把`code/configs/experiment.yaml`的`pretrained`改为`../v002_e16_v4e/weights/epoch180.pt`，`initialization_kind`改为`e16`，并明确设置本次微调轮数、学习率和save_epochs。它是新实验、新优化器，不能把v002的旧优化器状态作为v006续训。默认配置仍为官方初始化180轮，并未擅自选择微调。建版时的原E16输出一致性检查不等于已生成或已训练v006 PT。

## 本机运行

以下在项目根目录`C:\Users\16125\Desktop\钢材AI`执行；入口按自身位置定位，复用原环境，无需安装新环境。

```powershell
# 只检查数据和构图，使用合成输入，不训练、不推理真实评测图
& .\steel-defect-yolo\.venv\Scripts\python.exe .\yolo11-source-workspace\versions\v006_e16_p3p4_gate_v4e\code\scripts\train.py --check

# 显式开始正式训练；不附带推理
& .\steel-defect-yolo\.venv\Scripts\python.exe .\yolo11-source-workspace\versions\v006_e16_p3p4_gate_v4e\code\scripts\train.py --train

# 仅恢复本版本中断的训练，不能用于已完成任务或v002检查点
& .\steel-defect-yolo\.venv\Scripts\python.exe .\yolo11-source-workspace\versions\v006_e16_p3p4_gate_v4e\code\scripts\train.py --resume

# 训练后对明确指定的外部评测集执行V4E
& .\steel-defect-yolo\.venv\Scripts\python.exe .\yolo11-source-workspace\versions\v006_e16_p3p4_gate_v4e\code\scripts\predict.py --epoch 180 --source '.\复赛测试集\复赛发布' --dataset-name fusai
```

直接运行train.py而不传参数也只做检查。首次运行不覆盖已有训练。训练PT直接保存到weights/，每轮更新last.pt，默认额外保存160/170/180轮；模型、EMA、优化器、AMP及调度器状态保留。恢复校验代码、配置、数据和初始化权重指纹。恢复为epoch边界恢复，不保证多进程增强序列逐位相同。训练结束或失败自动刷新README和manifest；不自动启动推理。

V4E保持v002算法，默认最终导出过滤qilie，九类检测头不变。需要保留全部九类时，在predict命令末尾添加不带值的`--exclude-class`。不同数据集使用不同dataset-name；权重、源码、图像清单或过滤条件变化后，拒绝混用原结果缓存。改名后可重新运行。推理结束自动归档；框数不代表精度。

验证命令：`code/scripts/check_model.py`（合成输入，包括已有E16权重一致性）、`code/scripts/verify_release.py`（文件校验）。真实训练和评测均需要单独执行上述明确入口。

建版验证结果见[code/tests/VALIDATION.md](code/tests/VALIDATION.md)：9项单元测试、实际v002权重迁移恒等检查、合成检测损失反向、检查点状态恢复、CUDA/AMP 1024输入检查及全训练清单检查通过。完整真实数据训练循环尚未执行。

## 当前状态

<!-- RUN_STATUS_START -->
已完成训练轮数：0。尚未训练；weights/initialization/仅包含初始化权重。

尚无本版本平台评测成绩；不以训练损失、预测框数或结构检查代替精度。

尚未执行真实评测集V4E推理。
<!-- RUN_STATUS_END -->

## 已知限制

门控有效性尚需外部评测与消融确认。新增分支与门控均有零初始化层，早期部分梯度为零是预期现象，需检查连续更新。当前入口支持单GPU或CPU；完整数据、原缓存和GPU环境未打包。切换电脑需重新配置数据路径。源码和E16分支继承v002，训练入口参考本地后续版本的可恢复归档实现，未继承其数据划分。依赖和许可证见code/。
