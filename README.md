# NewsGuard-AI - 多模态虚假新闻检测系统

## 🎯 项目概述

NewsGuard-AI是一个基于深度学习的多模态虚假新闻检测系统，能够综合分析新闻的文本和图像内容，准确识别虚假新闻。系统采用先进的ISMAF架构，支持中英文双语检测。

### 核心特性

- **多模态融合分析**：同时处理文本和图像信息
- **双语支持**：基于PHEME（英文）和WEIBO（中文）数据集训练的双模型
- **可解释性分析**：提供文本注意力分布和图像Grad-CAM热力图
- **图文一致性评估**：计算文本与图像特征的余弦相似度
- **预测不确定性量化**：基于MC Dropout的熵值和方差估计
- **实时检测**：快速响应的Web界面
- **置信度评分**：提供可靠的检测置信度

## 🏗️ 系统架构

### 技术栈

**前端**
- React 18 + TypeScript
- Tailwind CSS
- Vite构建工具
- Lucide React图标库

**后端**
- FastAPI (Python)
- PyTorch深度学习框架
- 异步处理机制

## 🚀 快速开始

### 环境要求

**前端环境**
- Node.js >= 16.0.0
- npm >= 8.0.0 或 yarn >= 1.22.0

**后端环境**
- Python >= 3.8
- pip >= 21.0.0
- CUDA >= 11.0 (可选，用于GPU加速)

### 安装步骤

#### 1. 克隆项目

```bash
git clone <repository-url>
cd NewsGuard-AI
```

#### 2. 安装前端依赖

```bash
# 在项目根目录下执行
npm install
```

或使用 yarn:

```bash
yarn install
```

#### 3. 安装后端依赖

```bash
# 进入后端目录
cd backend

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

**注意**: 如果需要GPU加速，请根据你的CUDA版本安装对应的PyTorch版本：

```bash
# 例如 CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 启动运行

#### 启动后端服务

1. 确保已激活Python虚拟环境（如果使用）

2. 进入后端目录：
```bash
cd backend
```

3. 启动FastAPI服务器：
```bash
python main.py
```

或者使用uvicorn直接启动：
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

后端服务将在 `http://localhost:8000` 启动。

**重要提示**：
- 首次启动时，系统会自动加载中英文两个模型，可能需要一些时间
- 确保 `backend/checkpoint/` 目录下存在模型文件：
  - `mse-sin_en.pth` (英文模型)
  - `mse-sin_zh.pth` (中文模型)
- 确保 `backend/dataset/` 目录下存在必要的数据文件：
  - `vocab_en.pkl`, `vocab_zh.pkl` (词汇表)
  - `train_en.pkl`, `train_zh.pkl` (词嵌入矩阵)
  - `stopwords_en.txt`, `stopwords_zh.txt` (停用词)

#### 启动前端服务

1. 在项目根目录下执行：

```bash
npm run dev
```

或使用 yarn:

```bash
yarn dev
```

前端服务将在 `http://localhost:3000` 启动。

2. 在浏览器中打开 `http://localhost:3000` 访问应用

### 使用说明

1. **输入新闻文本**：在左侧输入框中输入待检测的新闻文本（支持中英文）

2. **上传图片**：点击上传区域选择新闻相关图片（**必填项**）

3. **执行分析**：点击"执行分析"按钮，系统将：
   - 自动检测文本语言
   - 加载对应的预训练模型
   - 进行多模态特征提取和融合
   - 返回检测结果和可解释性分析

4. **查看结果**：右侧面板将显示：
   - 检测结果（真实/虚假）及置信度
   - 图文一致性评分（余弦相似度）
   - 预测不确定性指标（熵值和方差）
   - 文本注意力权重分布
   - 图像Grad-CAM热力图

## 📁 项目结构

```
NewsGuard-AI/
├── backend/                 # 后端服务
│   ├── main.py             # FastAPI主程序
│   ├── model.py            # 模型定义(StudentNN, TextProcessor等)
│   ├── cam_explainer.py    # Grad-CAM可视化模块
│   ├── config_file.py      # 模型配置参数
│   ├── requirements.txt    # Python依赖
│   ├── checkpoint/         # 模型权重文件
│   │   ├── mse-sin_en      # 英文模型权重
│   │   └── mse-sin_zh      # 中文模型权重
│   └── dataset/            # 数据集文件
│       ├── vocab_en.pkl    # 英文词汇表
│       ├── vocab_zh.pkl    # 中文词汇表
│       ├── train_en.pkl    # 英文词嵌入矩阵
│       ├── train_zh.pkl    # 中文词嵌入矩阵
│       ├── stopwords_en.txt
│       └── stopwords_zh.txt
├── components/             # React组件
│   ├── FileUpload.tsx     # 文件上传组件
│   ├── ResultDashboard.tsx # 结果展示面板(含一致性/不确定性卡片)
│   └── TextHighlighter.tsx # 文本注意力高亮组件
├── services/               # 前端服务
│   └── detectionService.ts # API调用与数据映射
├── App.tsx                 # 主应用组件
├── types.ts                # TypeScript类型定义
├── index.tsx               # 应用入口
├── index.html              # HTML模板
├── package.json            # 前端依赖配置
├── vite.config.ts          # Vite构建配置
└── README.md               # 项目说明文档
```

## 🛠️ 开发说明

### 构建生产版本

**前端构建**:
```bash
npm run build
```

构建产物将输出到 `dist/` 目录

**预览生产构建**:
```bash
npm run preview
```

### API接口说明

**检测接口**: `POST /predict`

**请求参数**:
- `text` (string, required): 新闻文本内容
- `image` (file, required): 新闻相关图片

**响应格式**:
```json
{
  "detected_language": "zh",
  "prediction": "Fake",
  "confidence": 0.95,
  "probabilities": [0.05, 0.95],
  "consistency": {
    "cosine_similarity": 0.72
  },
  "uncertainty": {
    "entropy": 0.234,
    "variance": 0.012
  },
  "explainability": {
    "textAttention": [
      {"token": "新闻", "weight": 0.85},
      {"token": "报道", "weight": 0.62}
    ],
    "imageCamOverlay": "data:image/png;base64,..."
  }
}
```

**响应字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `detected_language` | string | 检测到的语言 (zh/en) |
| `prediction` | string | 预测结果 (Real/Fake) |
| `confidence` | float | 预测置信度 (0-1) |
| `probabilities` | array | 各类别概率分布 |
| `consistency.cosine_similarity` | float | 图文特征余弦相似度 (-1到1) |
| `uncertainty.entropy` | float | MC Dropout预测熵值 |
| `uncertainty.variance` | float | MC Dropout预测方差 |
| `explainability.textAttention` | array | 文本Token注意力权重列表 |
| `explainability.imageCamOverlay` | string | Grad-CAM热力图Base64编码 |

## 📝 注意事项

1. **图片为必填项**：系统需要同时分析文本和图像，图片上传是必须的

2. **模型加载时间**：首次启动或切换语言时，模型加载可能需要10-30秒

3. **内存要求**：建议至少8GB RAM，使用GPU时建议16GB以上

4. **浏览器兼容性**：推荐使用Chrome、Firefox、Edge等现代浏览器

## 📄 许可证

本项目为毕业设计项目，仅供学习和研究使用。