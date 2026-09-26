# 本地原图训练/测试划分 d001

原始数据：`C:\Users\16125\Desktop\钢材AI\train\train`。固定种子 42。原始图片和 XML 保留不变；此目录使用独立副本，未训练、未推理、尚未评测。

## 数量

| 集合 | 图片 | 有缺陷图片 | 无缺陷图片 | 去重、边界修正后框数 |
|---|---:|---:|---:|---:|
| all | 3200 | 2126 | 1074 | 5885 |
| train | 2560 | 1701 | 859 | 4707 |
| test | 640 | 425 | 215 | 1178 |

所有原图尺寸均为本次统计所列尺寸，详见 split_statistics.json。
原始 XML 合计 5889 个框；YOLO 转换删除 4 个完全重复框，修正 1 个越界框，保留 5885 个。归档 XML 未改写。

| 类别 | 全部图片 | 训练图片 | 测试图片 | 全部有效框 |
|---|---:|---:|---:|---:|
| gunyin | 236 | 189 | 47 | 253 |
| huashang | 60 | 48 | 12 | 153 |
| jiaza | 113 | 90 | 23 | 220 |
| jieba | 604 | 483 | 121 | 1190 |
| mamianmakeng | 309 | 247 | 62 | 2433 |
| qilie | 28 | 22 | 6 | 69 |
| yanghuatiepi | 358 | 286 | 72 | 492 |
| yiwuyaru | 464 | 371 | 93 | 602 |
| zonglie | 296 | 237 | 59 | 473 |

一张图片可能包含多种缺陷，类别图片数不能直接相加作为总图片数。

## 划分与校验

按原图划分，再处理裁块。首个 `-` 或 `_` 前的文件名前缀视为同来源组；完全相同 SHA256 图片也合组，两项取传递闭包。
完整组只进入一个集合；分组优化严格保证 2560:640，同时平衡类别阳性图片数、目标框数、背景图和两种文件名前缀分布。
分组前缀仅为保守代理，未经生产批次信息确认；未做感知近重复检测。随机种子和脚本可追溯；最终复现以 split_manifest.csv 为准（限时优化器跨版本可能得到其他等价分组）。
已核对图片/XML 配对、图像头尺寸、Pillow verify、标注有效性、复制文件 SHA256、YOLO 格式、集合完整覆盖、文件名/来源组/完全重复文件跨集零重叠。没有完整解码每张图片的像素。

## 目录与使用

- `images/train/`、`images/test/`：独立图片副本。
- `annotations/train/`、`annotations/test/`：原始 VOC XML。
- `labels/train/`、`labels/test/`：与原项目类别 ID 一致的 YOLO 标签；背景图为空 TXT。
- `data.yaml`：YOLO 数据入口；`train.txt`、`test.txt`：相对当前数据目录的清单。
- `split_manifest.csv`：固定划分和逐文件 SHA256；`manifest.json`：全部交付文件校验。
- `split_statistics.json`、`class_distribution.csv`、`source_audit.json`：统计与逐图审计。
- `code/split_dataset.py`：划分脚本；依赖 numpy、scipy、Pillow、PyYAML，复用项目现有环境。

原项目默认训练配置保持原样，尚未切换至本数据集。新实验应建立单独版本，并将数据入口设为本目录 data.yaml。
若训练使用裁块或增强，只能从 images/train 创建；旧全量裁块缓存包含测试原图，不能继续混用。
历史模型若训练过全部 3200 张图，此处 640 张对它们不构成独立测试集；可靠评估需在 2560 张上重新训练，不能从已见过全部原图的项目权重续训。
只有训练/测试两份，没有另建验证集。data.yaml 的 val 与 test 均指向这 640 张，仅用于兼容加载器；若用它调参、选 epoch、早停，就应称为验证集。保留最终测试用途时应从训练部分另划验证集，或关闭训练期验证并预先固定训练方案。
最终测试可用 `model.val(data=本目录data.yaml, split="test")`；本次没有启动模型调用，也没有任何精度分数。

重新生成到新目录（从项目根目录运行；脚本拒绝覆盖已存在划分）：

```powershell
& .\steel-defect-yolo\.venv\Scripts\python.exe .\yolo11-source-workspace\data\splits\d001_train_test_4to1\code\split_dataset.py --source .\yolo11-source-workspace\data\raw\train --output .\yolo11-source-workspace\data\splits\d001_train_test_4to1_rebuild --seed 42
```
