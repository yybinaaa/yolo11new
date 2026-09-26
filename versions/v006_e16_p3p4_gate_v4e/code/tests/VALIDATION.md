# v006建版验证

验证日期：2026-09-26。没有启动真实数据训练或真实评测集推理。

- `train.py --check`：实际训练器get_model构建官方初始化的九类门控模型；合成输入前向。全部3200张原图对应的15629条训练清单检查通过，图片检查存在性、大小和mtime，标签检查内容和SHA256。未创建数据划分。
- `check_model.py`：9项单元测试，包括非零E16残差恒等初始化、43,041新增参数、不同空间尺寸、forward_split和卷积融合、门控各层梯度、真实检测损失反向、训练器安装、完整模型序列化与重构、错误结构拒绝，以及合成模块检查点的live/EMA/优化器/调度器恢复与不匹配合约拒绝。
- 实际v002 epoch180 PT迁移后，CPU合成输入输出与父模型完全一致；Ultralytics加载临时PT后输出一致。
- RTX 4070 Laptop GPU上执行128与1024合成输入、CUDA FP16自动混合精度一致性与有限值检查。
- `check_inheritance.py`：逐文件比较继承的Ultralytics和原E16源码；比较V4E非入口函数AST，确认核心算法保持一致；核对官方初始化PT哈希。
- 版本文件完整性由`verify_release.py`检查，校验清单由`archive.py`生成。

测试中的合成梯度更新不代表真实训练完成。尚未执行完整数据训练循环、真实中断续训、真实V4E整套推理或平台评测。模型性能和实际训练显存尚未验证。

机器可读结果见`code/artifacts/model_check.json`、`data_check.json`、`verification.json`、`inheritance.json`。
