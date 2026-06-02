# 肺结节检测与分割系统软件工程化升级方案

## 目标

当前项目已经完成肺结节检测、分割、训练、评估和基础界面展示，但现有 PySide6 界面更偏算法验证 Demo。为了让项目更像完整的软件工程项目，建议将系统升级为一个具备现代化交互界面、后端推理服务、历史记录管理和实验结果展示能力的医学影像 AI 系统。

目标系统应覆盖：

- CT 图像上传与预览
- 检测模型和分割模型选择
- 一键运行肺结节检测与分割
- 检测框、置信度、分割 mask 可视化
- 历史病例记录管理
- 模型指标与消融实验结果展示
- 本地或服务器部署

## 推荐技术路线

推荐采用前后端分离架构：

```text
React / Vue 前端
        ↓
FastAPI 后端
        ↓
YOLO / U-Net 推理服务
        ↓
SQLite / 文件存储
```

相比继续扩展单窗口 PySide6，Web 架构更适合项目展示、远程部署、多人访问、界面美化和简历表达。

## 技术选型

| 模块 | 推荐技术 | 作用 |
|------|----------|------|
| 前端框架 | React + Vite + TypeScript | 构建现代化 Web 界面 |
| UI 组件库 | Ant Design / Arco Design | 表单、表格、按钮、弹窗、布局 |
| 图像交互 | Canvas / Konva.js / OpenSeadragon | CT 图像缩放、拖拽、检测框和 mask 叠加 |
| 图表展示 | ECharts / Recharts | 展示实验指标、模型对比结果 |
| 后端框架 | FastAPI | 上传图像、调用模型、返回推理结果 |
| 推理服务 | PyTorch + Ultralytics + OpenCV | 复用现有 YOLO / U-Net 推理流程 |
| 数据库 | SQLite / PostgreSQL | 保存病例、模型、推理历史和指标 |
| 文件存储 | uploads / outputs | 保存原图、叠加图、patch、mask、JSON |
| 部署 | Docker / docker-compose | 一键部署前后端与推理服务 |

## 推荐系统名称

可以将项目命名为：

- 基于深度学习的肺结节检测与分割辅助诊断系统
- 基于 React + FastAPI 的肺结节检测与分割系统
- 面向 CT 图像的肺结节智能分析平台

简历中推荐使用：

**基于 React + FastAPI 的肺结节检测与分割辅助诊断系统**

## 功能模块设计

### 1. 工作台首页

首页用于承载核心操作入口：

- 上传 CT 图像
- 选择检测模型
- 选择分割模型
- 一键开始分析
- 展示最近分析记录

示意：

```text
肺结节检测与分割系统

[上传 CT 图像] [选择检测模型] [选择分割模型] [开始分析]

最近病例：
时间 | 图像名 | 结节数量 | 最高置信度 | 操作
```

### 2. 图像分析页面

这是系统的核心页面。

建议布局：

```text
┌────────────────┬────────────────┬────────────────────┐
│ 原始 CT 图像    │ AI 分析结果      │ 结节列表 / 详情     │
│                │ 检测框 + mask   │ #1 conf=0.76       │
│ 支持缩放拖拽    │ 支持图层开关      │ 坐标、面积、patch    │
└────────────────┴────────────────┴────────────────────┘
```

交互能力：

- 原图与结果图并排查看
- 支持放大、缩小、拖拽和平移
- 支持显示 / 隐藏检测框
- 支持显示 / 隐藏分割 mask
- 点击某个结节时高亮对应检测框
- 展示置信度、框坐标、patch、mask 和 JSON 结果

### 3. 历史记录页面

保存每次分析记录，便于回看和演示。

字段建议：

- 分析时间
- 原图文件名
- 检测结节数量
- 最高置信度
- 使用的检测模型
- 使用的分割模型
- overlay 路径
- result.json 路径

功能建议：

- 搜索病例
- 按时间排序
- 查看详情
- 删除记录
- 重新分析

### 4. 模型管理页面

用于展示当前可用模型及其指标。

检测模型示例：

- `yolo11_luna16_full_best.pt`
- `full_yolo11n_baseline`
- `full_yolo11n_swin`
- `full_yolo11n_mspa`
- `full_yolo11n_swin_mspa`

分割模型示例：

- `unet_full_best.pt`
- 分割消融实验模型

展示指标：

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------|-----------|--------|---------|--------------|
| baseline | - | - | - | - |
| swin | - | - | - | - |
| mspa | - | - | - | - |
| swin+mspa | - | - | - | - |

分割模型展示：

| 模型 | DSC | IoU | SEN |
|------|-----|-----|-----|
| unet_full | - | - | - |

### 5. 消融实验结果页面

项目已经支持检测与分割消融实验，因此可以单独增加实验对比页面。

展示内容：

- 检测模型 ablation 指标表
- 分割模型 ablation 指标表
- mAP / Recall / DSC / IoU / SEN 柱状图
- 最佳模型高亮
- 支持下载 CSV

推荐图表：

- mAP@0.5 柱状图
- Recall 柱状图
- DSC / IoU / SEN 多指标柱状图
- Precision-Recall 对比图

## 后端 API 设计

建议使用 FastAPI 提供后端接口。

### 图像分析接口

```http
POST /api/analyze
```

请求：

- 图像文件
- 检测权重名称或路径
- 分割权重名称或路径
- 置信度阈值

响应：

```json
{
  "case_id": "20260602_001",
  "image_path": "uploads/case.png",
  "overlay_path": "outputs/case/overlay.png",
  "detections": [
    {
      "x1": 207,
      "y1": 401,
      "x2": 224,
      "y2": 417,
      "confidence": 0.54
    }
  ],
  "patch_paths": ["outputs/case/patch_00.png"],
  "mask_paths": ["outputs/case/mask_00.png"],
  "mode": "model"
}
```

### 历史记录接口

```http
GET /api/history
GET /api/history/{case_id}
DELETE /api/history/{case_id}
```

### 模型列表接口

```http
GET /api/models/detection
GET /api/models/segmentation
```

### 实验指标接口

```http
GET /api/metrics/detection
GET /api/metrics/segmentation
```

## 推荐项目结构

```text
lung-nodule/
├── backend/
│   ├── main.py
│   ├── api/
│   │   ├── analyze.py
│   │   ├── history.py
│   │   ├── models.py
│   │   └── metrics.py
│   ├── services/
│   │   ├── inference_service.py
│   │   ├── model_registry.py
│   │   └── storage_service.py
│   ├── database/
│   │   ├── models.py
│   │   └── session.py
│   ├── uploads/
│   └── outputs/
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Analyze.tsx
│   │   │   ├── History.tsx
│   │   │   ├── Models.tsx
│   │   │   └── Experiments.tsx
│   │   ├── components/
│   │   │   ├── ImageViewer.tsx
│   │   │   ├── DetectionOverlay.tsx
│   │   │   ├── ModelSelector.tsx
│   │   │   └── MetricsChart.tsx
│   │   ├── api/
│   │   └── main.tsx
│   └── package.json
│
├── models/
├── configs/
├── scripts/
├── weights/
├── docker-compose.yml
└── README.md
```

## 与现有代码的关系

现有代码不需要全部推翻，可以按以下方式复用：

| 现有模块 | 新系统中的作用 |
|----------|----------------|
| `pipeline.py` | 封装为后端推理服务 |
| `models/segmentation/` | 继续作为 U-Net 模型定义 |
| `models/detection/` | 继续作为 YOLO 自定义模块 |
| `eval_detection.py` | 指标生成脚本 |
| `eval_segmentation.py` | 分割指标生成脚本 |
| `outputs/` | 推理结果存储 |
| `runs/` | 训练权重与实验结果来源 |

迁移策略：

1. 保留当前训练、评估和推理代码。
2. 将 `pipeline.run_pipeline()` 封装成 FastAPI 服务。
3. 前端通过 HTTP 调用后端接口。
4. 历史记录从当前 SQLite 逻辑升级为统一病例表。
5. 实验结果从 CSV / results.csv 读取并可视化。

## 开发路线

### 第一阶段：后端服务化

- 新建 `backend/`
- 使用 FastAPI 封装 `/api/analyze`
- 支持上传图片并返回 overlay、detections、mask 路径
- 复用当前 `pipeline.py`

### 第二阶段：前端基础界面

- 新建 `frontend/`
- 使用 React + Vite + TypeScript
- 搭建首页、上传组件、结果展示页面
- 支持图像上传和结果查看

### 第三阶段：增强可视化交互

- 使用 Canvas / Konva.js 绘制检测框
- 支持 mask 半透明叠加
- 支持缩放、拖拽、结节列表联动高亮

### 第四阶段：历史记录和模型管理

- 增加病例历史记录
- 增加模型列表和模型指标展示
- 支持选择不同检测 / 分割权重推理

### 第五阶段：实验对比页面

- 读取 ablation CSV
- 绘制指标图表
- 高亮最佳模型
- 支持 CSV 下载

### 第六阶段：部署

- 编写 Dockerfile
- 编写 docker-compose
- 前后端一键启动
- README 加入部署和演示说明

## 简历表述升级

升级后可写成：

**基于 React + FastAPI 的肺结节检测与分割辅助诊断系统**

项目亮点：

- 深度学习医学影像检测与分割
- YOLO / U-Net 模型训练与消融实验
- 前后端分离架构
- GPU 推理服务封装
- 医学图像可视化交互
- SQLite / 文件系统结果管理
- Docker 化部署

这比单纯 PySide6 Demo 更能体现软件工程能力，也更适合简历和项目答辩展示。

