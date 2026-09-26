共享路径已于 2026-09-26 更新：数据见仓库 `data/`；初始权重见 `pretrained/`；当前生成数据位于 `data/generated/v001_baseline/`。旧路径记录保留用于溯源。详见 [路径重构说明](../../SHARED_LAYOUT.md)。

# v001_baseline

这是从当前可运行环境复制出的 Ultralytics 8.4.98 源码基线，保留为后续实验的父版本。

- `ultralytics/`：完整可执行 Python 包源码、内置配置和资源。
- `models/yolo11n.yaml` 至 `models/yolo11x.yaml`：YOLO11 五种尺度的独立入口文件。
- `LICENSE`：上游 AGPL-3.0 许可证。

不要直接修改基线。先运行 `scripts/new_version.py` 创建新版本，再修改新目录中的网络层、任务构建代码或模型 YAML。

## 自动批量推理复赛测试集

新增本版本独立入口 `START_INFERENCE.cmd` / `INFER_FUSAI.py`，自动读取本版本weights中的所有PT，依次运行V4E，输出至本版本inference_packages/fusai_auto。使用方法与断点缓存说明见 [INFERENCE_README.md](INFERENCE_README.md)。本次仅添加代码，未执行复赛批量推理。
