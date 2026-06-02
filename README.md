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
| 检测 | Params(M) | 模型参数量，评估脚本自动输出 |
| 分割 | DSC(%) / IoU(%) / SEN(%) / ASD(mm) | 91.20 / 85.40 / 91.70 / 需真实 spacing |

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

**检测消融（不加入 KAN-C3k2）**：

```bash
# 单个实验
python train_detection_ablation.py --experiment baseline
python train_detection_ablation.py --experiment swin
python train_detection_ablation.py --experiment mspa
python train_detection_ablation.py --experiment swin_mspa

# 按顺序训练：YOLOv11n -> +Swin -> +MSPA -> +Swin+MSPA
python train_detection_ablation.py --experiment all
```

当前消融顺序：

```text
YOLOv11n baseline -> YOLOv11n + SwinTinyLayer -> YOLOv11n + MSPA -> YOLOv11n + SwinTinyLayer + MSPA
```

最后一个模型可作为 `SKM-YOLO-lite / w/o KAN-C3k2`，不要称为完整 SKM-YOLO。

**分割**：

```bash
python train_segmentation.py \
  --data datasets/segmentation \
  --output runs/segmentation/unet_full \
  --epochs 100 --batch 32
```

权重输出：`runs/segmentation/unet_full/best.pt`

**分割消融**：

```bash
# 单个实验
python train_segmentation_ablation.py --experiment baseline
python train_segmentation_ablation.py --experiment fdconv
python train_segmentation_ablation.py --experiment fdconv_capsule
python train_segmentation_ablation.py --experiment fdconv_rfapm
python train_segmentation_ablation.py --experiment full

# 按顺序训练：U-Net -> +FDConv -> +FDConv+Capsule -> +FDConv+RF-APM -> full
python train_segmentation_ablation.py --experiment all
```

当前消融顺序：

```text
U-Net baseline -> U-Net + FDConv -> U-Net + FDConv + Capsule path -> U-Net + FDConv + RF-APM -> Caps-FDRNet-lite
```

`full` 对应 `caps_fdrnet_lite`，是按论文方向实现的轻量复现版；在没有完全对齐论文细节前，不建议称为完整 Caps-FDRNet。

## 评估

```bash
python eval_detection.py \
  --weights runs/detection/yolo11_luna16_full/weights/best.pt

python eval_detection_ablation.py \
  --project runs/detection_ablation \
  --output runs/detection_ablation/metrics.csv

python eval_segmentation.py \
  --weights runs/segmentation/unet_full/best.pt \
  --spacing-x 1.0 --spacing-y 1.0

python eval_segmentation_ablation.py \
  --project runs/segmentation_ablation \
  --spacing-x 1.0 --spacing-y 1.0 \
  --output runs/segmentation_ablation/metrics.csv
```

检测评估输出：**Precision(P)**、**Recall(R)**、**mAP@0.5**、**mAP@0.5:0.95**、**Params(M)**。

分割评估输出：**DSC(%)**、**IoU(%)**、**SEN(%)**（敏感度 / 像素级召回）、**ASD(mm)**。ASD 依赖像素间距；当前 patch 数据未保存真实 spacing 时，默认按 `1.0 mm/pixel` 计算，接入 DICOM / LIDC 真实 spacing 后应传入真实值。

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
├── train_detection_ablation.py # 检测消融训练（baseline / Swin / MSPA / Swin+MSPA）
├── train_segmentation.py   # U-Net 训练
├── train_segmentation_ablation.py # 分割消融训练（U-Net / FDConv / Capsule / RF-APM / full）
├── eval_detection.py
├── eval_detection_ablation.py
├── eval_segmentation.py    # DSC(%) / IoU(%) / SEN(%) / ASD(mm)
├── eval_segmentation_ablation.py
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
- 分割 ASD 使用真实体素 spacing（需在预处理阶段保留并传入评估）
- 论文方法：SKM-YOLO、Caps-FDRNet 等

## 许可证

请遵守 LUNA16 / LIDC 数据使用协议；代码仅供学习与研究使用。
