# 共享数据与权重：2026-09-26 路径重构

适用范围：v001～v005。v006 暂不处理。Python 环境、依赖安装、模型结构、超参数、数据划分和已训练权重不变。

共享数据见 [data/README.md](data/README.md)，官方初始化权重见 [pretrained/README.md](pretrained/README.md)。

## v005 当前入口

以下在仓库根目录运行，python 指已安装本项目依赖的解释器；本机 PowerShell 启动脚本仍保留原环境默认值。

```text
# 只检查原始划分及文件校验，不训练、不生成切片
python versions/v005_yolo11m_v4e_split/code/scripts/prepare_data.py --check --verify-images
# 只检查推理输入、权重及模型加载，不做图片预测
python versions/v005_yolo11m_v4e_split/INFER_FUSAI.py --check
```

数据准备输出由版本名决定：`data/generated/<版本名>/`。训练记录和推理结果仍按版本归档。
批量复赛推理默认输入 `data/fusai/`，可用 --source 指定其他图片目录。
code/scripts/predict.py 的本地评测入口仍严格验证固定划分；只用自己的图片推理时使用 INFER_FUSAI.py。

## 新建后续实验

```text
python scripts/new_version.py v007_yolo11m_example --from-version v005_yolo11m_v4e_split --description "实验目的和模型改动"
```

该命令只创建新源码版本，不启动数据准备或训练。示例名称不是此次创建的实际版本。
默认从 v005 创建，也支持 v004 和中央配置中的基线版本；v002/v003 仅有历史训练归档，v006 尚未迁移，因此不会被错误当作可直接复制的训练模板。
新版本共享 data/splits/ 和 pretrained/，拥有独立 data/generated/<版本名>/、weights/、训练记录和推理结果。
创建时不复制父版本 PT、旧训练记录、缓存或推理结果。Ultralytics 的 data 源码包仍完整复制。
在新版本目录按其 README 的检查→准备→训练命令运行。

## 历史记录与继续训练

原始数据、旧 datasets/、旧 code/data/、旧 code/pretrained/ 与已有 PT、训练记录、推理结果均保留。
本次修改前的源码、配置和索引备份位于 artifacts/path_migration_20260926/original_files/。
路径重构改变了源码与缓存校验值，不能把新源码伪装成旧训练过程继续运行：旧 contract.json 和 PT 中的训练合同未被改写。
v005 已完成 180 轮，保留的 PT 可继续加载用于推理；需要新训练请创建新版本。
如果需要恢复重构前中断的训练，必须先恢复对应完整代码和数据布局，不能绕过合同校验。
新创建版本自身的恢复训练会显式采用当前位置的数据配置，避免 PT 内的旧绝对路径重新指回其他目录。
已有推理输出合同保留原样；需要再次推理时应使用新的输出分组，避免与旧记录混合。

运行时 Ultralytics 自动生成的日志可能记录绝对路径；它们是历史记录，不是可移植配置。
本次验证仅做文件校验、合成测试、模型加载与路径检查，没有正式训练或真实图片预测。
