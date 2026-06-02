# 肺结节检测分割 MVP

这是基于论文《基于深度学习的 CT 图像肺部结节检测与分割研究》的两天简化版工程骨架。

目标不是完整复现 SKM-YOLO 和 Caps-FDRNet，而是先跑通：

```text
CT 图像 -> 检测框 -> 64x64 patch -> 分割 mask -> 叠加可视化 -> 简单界面
```

## 方案调整

原实施方案的阶段 A 写的是“两天 MVP 不含界面”。本版本按你的新目标加入了一个轻量 PySide6 界面。为了保证两天内可演示：

- 有 YOLO / U-Net 权重时，优先走真实模型推理。
- 没有权重时，自动走启发式检测与 Otsu 分割，只用于界面和流程 demo。
- LUNA16 的分割 mask 在 MVP 中用检测框椭圆近似生成；严格分割应在后续接入 LIDC-IDRI XML 轮廓。

## 环境

```bash
conda create -n lung-nodule python=3.9
conda activate lung-nodule
pip install -r requirements.txt
```

如果服务器是 RTX 4090 / 4090D 或较新的 vGPU，建议按服务器驱动安装合适的 PyTorch CUDA wheel。

## 数据准备

推荐先放 LUNA16 的 `subset0`、`subset1`、`subset2` 和 `annotations.csv`：

```text
data/LUNA16/
  subset0/
  subset1/
  subset2/
  annotations.csv
```

注意：LUNA16 实际常见目录是 `subset0` 到 `subset9`，标注文件通常叫 `annotations.csv`。

## 预处理

```bash
python scripts/preprocess_luna16.py --data-dir data/LUNA16 --annotations data/LUNA16/annotations.csv --output-dir datasets
python scripts/build_seg_patches.py --detection-dir datasets/detection --output-dir datasets/segmentation
```

## 训练

检测 baseline：

```bash
python train_detection.py --data configs/luna16_subset.yaml --model yolo11n.pt --epochs 80 --batch 8
```

分割 baseline：

```bash
python train_segmentation.py --data datasets/segmentation --epochs 100 --batch 32
```

## 评估

```bash
python eval_detection.py --weights runs/detection/yolo11_luna16_mvp/weights/best.pt
python eval_segmentation.py --weights runs/segmentation/unet_mvp/best.pt
```

## 命令行推理

无权重 demo：

```bash
python pipeline.py path/to/ct_slice.png --output-dir outputs/demo
```

使用训练权重：

```bash
python pipeline.py path/to/ct_slice.png --det-weights runs/detection/yolo11_luna16_mvp/weights/best.pt --seg-weights runs/segmentation/unet_mvp/best.pt
```

输出包括：

- `overlay.png`
- `patch_*.png`
- `mask_*.png`
- `result.json`

## 简单系统界面

```bash
python -m app.main
```

界面支持：

- 选择 PNG/JPG/TIFF/DICOM/MHD。
- 可选填写检测和分割权重。
- 运行检测 + 分割。
- 显示叠加图、检测框列表。
- SQLite 保存历史记录到 `outputs/history.sqlite`。

## 两天执行建议

Day 1：

- 准备 LUNA16 subset0~2。
- 跑 `preprocess_luna16.py` 和 `build_seg_patches.py`。
- 训练 YOLOv11n 50~80 epoch。
- 用界面验证上传和叠加图流程。

Day 2：

- 训练 U-Net 80~100 epoch。
- 把训练权重接入 `pipeline.py` 和界面。
- 保存 2~3 个成功样例与失败样例，写入复现报告。

## 后续扩展

- 用 LIDC-IDRI XML 生成真实分割 mask。
- 替换 YOLOv11 为 SKM-YOLO。
- 替换 U-Net 为 Caps-FDRNet。
- 增加 DSC、IoU、ASD、误差传播分析和更完整的历史记录管理。
