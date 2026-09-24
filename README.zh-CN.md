# CuPGeo 论文核心实验代码

本目录是 2026-09-22 整理的独立精简包，对应当前修订稿。未来发布时，请把 **`release_20260922` 本身**作为仓库根目录；上层 `icasspcode` 的旧代码、结果与说明仍保留，未覆盖。

## 已整理的代码

| 内容 | 文件/目录 |
| --- | --- |
| CP 杯保持嵌套、SG、VRA 有界校正 | `c3tta/models/multitask.py` |
| DINOv3、LoRA、Pyramid-FPN | `c3tta/models/backbones.py`、`adapters.py`、`decoder.py` |
| 分割、VRA、矩比例与固定辅助损失 | `c3tta/losses/` |
| 数据转换、标注与双视图增强 | `c3tta/data/`、`scripts/prepare_*.py` |
| 源域训练与两阶段初始化 | `scripts/train_source.py` |
| 无标签冻结预测与统一离线评分 | `scripts/evaluate.py`、`score_predictions.py` |
| 完整模型、四项组件消融、SG、权重配置 | `configs/` |
| 比例代理、直径误差、概率分析 | `scripts/analyze_ratio_geometry.py` |
| 多 seed 命令计划与原有汇总工具 | `scripts/plan_experiments.py`、`aggregate_source_method_seeds.py` |
| 几何/梯度测试与发布检查 | `tests/`、`scripts/check_release.py` |

模型、损失和训练实现保留原有 `c3tta` 包名及 checkpoint 格式。主方法没有测试时自适应。训练科学代码未重写；新增的是配置、运行计划和发布检查。

## 先区分两种训练协议

- **单阶段主对比**：每次 120 epochs；在单独的 `single/` 目录运行，不把两阶段结果作为这组运行的结果。
- **组件及 SG 消融**：先 CP 预训练 120 epochs，各变体再从相同 seed 的 CP `best.pt` 独立训练 120 epochs。CP-only 也续训第二阶段。关闭 SG 只发生在第二阶段。

查看命令，不会启动训练：

```bash
python -m scripts.plan_experiments --suite single --seeds 0 1 2
python -m scripts.plan_experiments --suite matched --seeds 0 1 2
```

只有显式加 `--execute` 才会依次执行。默认 matched 计划为 3 次 CP 初始化 + 15 次第二阶段训练，包含 SG 对照。两种预算的配置与结果应分别追溯；本次只整理代码，没有重跑论文数据。

## 环境、数据与运行

在独立 Python 3.10+ 环境安装与硬件对应的 PyTorch/torchvision，然后：

```bash
python -m pip install -e '.[dev,analysis]'
python -m pytest -q
```

安装、训练、预测和评分的完整命令见 [English README](README.md)。数据准备见 [DATA.md](docs/DATA.md)。数据集、DINOv3 预训练权重及实验 checkpoint 需另行准备，不包含在此目录。

默认设置为 REFUGE 320/80、768×768、FP16、batch 4、梯度累积 4、seed 0/1/2。源域验证 Mean Dice 选 checkpoint，目标域只评估。先冻结预测，再加载标签评分；OD/OC 共享阈值 0.5。四域先在每个 seed 内等权平均，再计算跨 seed 均值和样本标准差。

## 发布范围与验证

- 已加入 w/o VRA 和第二阶段 w/o SG 配置；共同计分器及比例分析取自后续实验归档。
- 没有加入真实图像、mask、权重、运行日志、逐图结果、服务器账户或远程调度代码。
- 发布内容仅限源码与说明，不包含各 seed 的实验权重、日志、预测或结果。
- 本目录源码采用 [MIT 许可证](LICENSE)；数据集、预训练权重及第三方依赖分别遵守各自条款。
- 检查范围及未完成的全量复现见 [VALIDATION.md](docs/VALIDATION.md)。论文结果不能因代码整理或合成测试通过而视为重新复现。
