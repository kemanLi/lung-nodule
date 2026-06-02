# 肺结节检测分割 MVP

基于论文《基于深度学习的 CT 图像肺部结节检测与分割研究》的简化工程：在 LUNA16 上跑通 **检测（YOLOv11n）→ 64×64 patch → 分割（U-Net）→ 可视化 → PySide6 界面**。

目标不是完整复现 SKM-YOLO / Caps-FDRNet，而是先建立可复现的 baseline，再逐步改进。

```text
CT 图像 → 检测框 → 64×64 patch → 分割 mask → 叠加图 → 界面 / 命令行
```

## 预训练权重（推荐）

全量数据 baseline 权重已发布在 [Releases v1.0-full](https://github.com/kemanLi/lung-nodule/releases/tag/v1.0-full)：

| 文件 | 模型 | 大小 |
|------|------|------|
| `yolo11_luna16_full_best.pt` | YOLOv11n 检测 | ~5 MB |
| `unet_full_best.pt` | U-Net 分割 | ~30 MB |

下载后放到任意目录，推理时通过 `--det-weights` / `--seg-weights` 指定路径即可。

**验证集指标**（val = subset8，与 Release 说明一致）：

| 任务 | 指标 | 数值 |
|------|------|------|
| 检测 | mAP@0.5 | 0.743 |
| 检测 | mAP@0.5:0.95 | 0.365 |
| 检测 | Precision / Recall | 0.729 / 0.734 |
| 分割 | DSC / IoU / SEN | 0.912 / 0.854 / 0.917 |

> 分割真值为检测框生成的**椭圆近似**，非 LIDC 医生轮廓，指标不宜与论文数值直接横比。

## 环境

```bash
conda create -n lung-nodule python=3.9
conda activate lung-nodule
pip install -r requirements.txt
```

- 推荐 **Python 3.9**（`lung-nodule` 环境已验证）。
- GPU 训练/推理：按 [PyTorch 官网](https://pytorch.org/) 安装带 CUDA 的 wheel（如 RTX 4090 + cu128）。

## 数据准备

将 LUNA16 原始数据放到：

```text
data/LUNA16/
  subset0/ … subset9/    # 每个 subset 下为 .mhd + .raw
  annotations.csv
  candidates.csv         # 可选
```

当前预处理脚本的**默认划分**（`scripts/preprocess_luna16.py`）：

| 子集 | 用途 |
|------|------|
| subset0 ~ subset7 | train |
| subset8 | val |
| subset9 | test |

## 预处理

```bash
python scripts/preprocess_luna16.py \
  --data-dir data/LUNA16 \
  --annotations data/LUNA16/annotations.csv \
  --output-dir datasets

python scripts/build_seg_patches.py \
  --detection-dir datasets/detection \
  --output-dir datasets/segmentation
```

预处理会生成 `datasets/detection`（YOLO 格式 PNG + 标签）和 `datasets/segmentation`（64×64 patches + masks）。

## 训练

**检测**（注意 `--project` 使用**绝对路径**，避免 ultralytics 将权重写到 `runs/detect/runs/detection/...` 嵌套目录）：

```bash
python train_detection.py \
  --data configs/luna16_subset.yaml \
  --model yolo11n.pt \
  --epochs 80 --batch 8 \
  --project /path/to/lung-nodule-mvp/runs/detection \
  --name yolo11_luna16_full
```

权重输出：`runs/detection/yolo11_luna16_full/weights/best.pt`

**分割**：

```bash
python train_segmentation.py \
  --data datasets/segmentation \
  --output runs/segmentation/unet_full \
  --epochs 100 --batch 32
```

权重输出：`runs/segmentation/unet_full/best.pt`

## 评估

```bash
python eval_detection.py \
  --weights runs/detection/yolo11_luna16_full/weights/best.pt

python eval_segmentation.py \
  --weights runs/segmentation/unet_full/best.pt
```

分割评估输出：**DSC**、**IoU**、**SEN**（敏感度 / 像素级召回）。

## 命令行推理

无权重（启发式 demo，仅流程演示）：

```bash
python pipeline.py path/to/ct_slice.png --output-dir outputs/demo
```

使用训练或 Release 权重：

```bash
python pipeline.py path/to/ct_slice.png \
  --det-weights path/to/yolo11_luna16_full_best.pt \
  --seg-weights path/to/unet_full_best.pt \
  --output-dir outputs/demo
```

支持输入：PNG/JPG/TIFF、DICOM（`.dcm`）、MetaImage（`.mhd` 取中间层）。

输出目录包含：

- `overlay.png` — 检测框 + 分割叠加（黄框=检测，红区=分割，数字=置信度）
- `patch_*.png` / `mask_*.png`
- `result.json`

## 图形界面（本地）

需有桌面环境（Windows / macOS 或带显示的 Linux）：

```bash
python -m app.main
```

功能：选择图像、填写检测/分割权重路径、运行推理、查看叠加图与检测列表；历史记录保存在 `outputs/history.sqlite`。

## 项目结构

```text
lung-nodule-mvp/
├── app/                    # PySide6 GUI
├── configs/luna16_subset.yaml
├── models/segmentation/    # U-Net 定义
├── scripts/                # LUNA16 预处理、patch 构建
├── pipeline.py             # 端到端推理
├── train_detection.py      # YOLO 训练
├── train_segmentation.py   # U-Net 训练
├── eval_detection.py
├── eval_segmentation.py    # DSC / IoU / SEN
├── data/                   # 原始 LUNA16（git 忽略）
├── datasets/               # 预处理输出（git 忽略）
└── runs/                   # 训练权重与曲线（git 忽略）
```

## 推理模式说明

| 条件 | 行为 |
|------|------|
| 提供 `--det-weights` 且文件存在 | YOLO 检测 |
| 未提供检测权重 | Otsu + 轮廓启发式检测（demo） |
| 提供 `--seg-weights` 且文件存在 | U-Net 分割 |
| 未提供分割权重 | patch 上 Otsu 分割（demo） |

## 后续改进方向

- 修复预处理中结节贴边导致的负坐标标注（`make_label` clamp）
- 接入 LIDC-IDRI XML 真实分割轮廓
- 更大检测模型（yolo11s/m）、数据增强、调 `--conf`
- 分割指标补充 ASD（需保留体素 spacing）
- 论文方法：SKM-YOLO、Caps-FDRNet 等

## 许可证

请遵守 LUNA16 / LIDC 数据使用协议；代码仅供学习与研究使用。
