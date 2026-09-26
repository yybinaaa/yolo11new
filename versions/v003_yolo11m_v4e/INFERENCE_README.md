共享路径已于 2026-09-26 更新：数据见仓库 `data/`；初始权重见 `pretrained/`；当前生成数据位于 `data/generated/v003_yolo11m_v4e/`。旧路径记录保留用于溯源。详见 [路径重构说明](../../SHARED_LAYOUT.md)。

# 本版本复赛批量推理

双击 `START_INFERENCE.cmd`，或在本版本目录执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\START_INFERENCE.ps1"
```

入口以自身目录为准，自动读取本版本 `weights/*.pt`，默认测试集为仓库根目录的 `data/fusai/`（当前788张）。每个权重在独立进程中依次推理，释放显存后再加载下一个。不扫描其他版本或官方初始化目录。

v005 默认运行 best.pt、epoch160.pt、epoch170.pt、epoch180.pt、last.pt 五份文件；last与epoch180对应同轮，默认仍分别输出。可指定四份以避免重复计算：

```powershell
.\START_INFERENCE.ps1 -Weights best.pt,epoch160.pt,epoch170.pt,epoch180.pt
# 只检查模型加载、类别、网络分支、固定划分来源；不推理图片
.\START_INFERENCE.ps1 -CheckOnly
# 另存标注图片
.\START_INFERENCE.ps1 -Visualize
# 保留qilie，变更设置后结果自动保存到另一个签名目录
.\START_INFERENCE.ps1 -KeepQilie
# 指定其他源目录或Python环境
.\START_INFERENCE.ps1 -Source "D:\复赛发布" -PythonExe "D:\env\python.exe"
```

也可直接运行 `python INFER_FUSAI.py`。参数包括 `--check`、`--weights best.pt epoch160.pt`、`--source 路径`、`--keep-qilie`、`--visualize`、`--device 0`。

## 输出与恢复

```text
本版本/
├─ INFER_FUSAI.py
├─ START_INFERENCE.cmd
├─ START_INFERENCE.ps1
├─ inference_tools/                 独立推理源码及Windows运行支持
└─ inference_packages/fusai_auto/
   ├─ batch_status.json
   ├─ best/<运行签名>/
   ├─ epoch160/<运行签名>/
   ├─ epoch170/<运行签名>/
   ├─ epoch180/<运行签名>/
   └─ last/<运行签名>/
```

每份结果包含 `predictions.json`（包括零检测图片）、`submission.json`、`submission.zip`（内部仅submission.json）、`summary.json`、`checksums.json`、`contract.json`、`progress.json`、逐图 `cache/`，可选 `visualizations/`。

运行签名绑定权重字节、图片内容、推理与模型源码、设备、类别过滤和可视化设置。重复相同命令自动复用已完成图片，变化后进入新目录，保留旧结果。一个权重失败会记录失败并继续其他权重，最后返回失败状态；修复后重跑同一命令即可。请保持窗口打开。

沿用项目归档的V4E流程：1024/20%切片+整图、自适应重切片、弱证据确认与safe-add。默认沿用已有复赛提交口径排除qilie，`-KeepQilie`可保留9类。不调整模型结构、训练代码或训练配置。E16版本使用自身源码加载自定义模块；新划分版本检查checkpoint内的划分指纹。

版本没有weights/PT时明确提示并退出，不会下载模型或自动训练。v001是源码基线，v004当前尚无训练权重；为两者也预置了相同入口。每个版本均可独立运行，不依赖其他版本的推理脚本。

完成或失败后更新本版本README状态及manifest中的新增代码和推理结果清单；缓存和可视化不计入版本manifest，主输出另有checksums.json。复赛图片无标签，本入口不计算准确率；检测框数量不代表精度。脚本仅生成本地提交包，不自动上传。

## 本次验证

2026-09-26：5项合成测试通过；v002的3份E16权重、v003的3份YOLO11m权重、v005的5份新划分权重均成功加载并通过检查；v001/v004无权重时正常提示退出。检查未对复赛图片运行检测。
