# 共享数据目录

所有路径均以仓库根目录 `yolo11-source-workspace/` 为基准。

```text
data/
├─ raw/train/                         # 3200 张原训练图片及对应 XML
├─ fusai/                             # 复赛测试图片
├─ splits/d001_train_test_4to1/
│  ├─ images/train/                   # 原有固定划分：2560 张
│  ├─ images/test/                    # 原有固定划分：640 张
│  ├─ labels/train/、labels/test/      # YOLO 标注
│  ├─ annotations/                    # 原 XML 标注
│  ├─ split_manifest.csv              # 原划分清单，SHA256 不变
│  └─ data.yaml                       # 相对本文件位置解析
└─ generated/
   ├─ v001_baseline/                  # 旧基线已有的全量训练切片副本
   ├─ v005_yolo11m_v4e_split/          # 原 v005 缓存的独立副本及相对加载清单
   └─ <新版本名称>/                    # 由相应版本 prepare_data.py 生成
```

本次复制并校验数据，没有重新随机划分、训练或执行模型推理。原始位置及历史归档保留。
640 张 test 在已有训练中用于每轮验证，是开发验证集，不是全新独立测试集。
原训练集和划分集为独立文件副本；以后运行 prepare_data.py 时可能在缓存中创建硬链接，勿手工编辑缓存图片或标注。

YAML 不保存 `C:/Users/...` 或 `path: .`：省略 path，让加载器以绝对传入的 YAML 文件所在目录为根。
train.txt/val.txt 采用 `./tiles/...`、`./full/...` 等相对路径，移动整个仓库后仍可加载。
版本输出仍存放在各自 weights/、code/training_records/ 和 inference_packages/，不混入共享原始数据。

v004/v005 训练配置中的 dataset 以版本目录为基准；共享初始权重在 ../../pretrained/yolo11m.pt。
此轮尚未改动 v006；它仍使用自己的原配置。

训练数据和大文件由 .gitignore 排除，不会自动进入 Git。将来发布代码时仍须单独提供数据和权重的获取方式。
