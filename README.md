# NewsGuard-AI - 多模态虚假新闻检测系统

## 🎯 项目概述

NewsGuard-AI是一个基于深度学习的多模态虚假新闻检测系统，通过综合分析新闻的文本和图像内容，实现虚假新闻的精准识别。系统采用先进的ISMAF架构，支持中英双语检测，并提供完整的可解释性分析。

### 核心特性

- **多模态融合分析**：同时处理文本和图像信息，提升检测准确性
- **双语支持**：基于PHEME（英文）和WEIBO（中文）数据集训练的双模型架构
- **可解释性分析**：提供文本注意力分布和Grad-CAM图像热力图
- **图文一致性评估**：计算文本与图像特征的余弦相似度，识别不匹配内容
- **预测不确定性量化**：基于MC Dropout的熵值和方差估计
- **实时检测**：快速响应的Web界面，即时返回分析结果
- **置信度评分**：提供可靠的检测置信度指标

## 🏗️ 系统架构

### 技术栈

**前端**
- React 19 + TypeScript
- Tailwind CSS
- Vite构建工具
- Lucide React图标库

**后端**
- FastAPI (Python)
- PyTorch深度学习框架
- 异步处理机制
- OpenCV图像处理

## 🚀 快速开始

### 环境要求

**前端**: Node.js >= 16.0.0, npm >= 8.0.0 或 yarn >= 1.22.0

**后端**: Python >= 3.8, pip >= 21.0.0, CUDA >= 11.0 (可选GPU加速)

### 安装步骤

```bash
# 克隆并进入项目
git clone <repository-url>
cd NewsGuard-AI

# 安装前端依赖
npm install

# 安装后端依赖
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 启动运行

**后端服务**:
```bash
cd backend
python main.py
# 或 uvicorn main:app --host 0.0.0.0 --port 8000
```
后端运行在 `http://localhost:8000`

**前端服务**:
```bash
npm run dev
```
前端运行在 `http://localhost:3000`

**注意**:
- 首次启动自动加载中英文模型（约10-30秒）
- 确保 `backend/checkpoint/` 和 `backend/dataset/` 目录文件完整

### 使用说明

1. **输入文本**：在输入框中输入新闻文本（支持中英文）
2. **上传图片**：点击上传区域选择新闻配图（必填）
3. **执行分析**：点击分析按钮，系统自动检测语言并返回结果
4. **查看结果**：右侧面板显示检测结果、置信度、图文一致性、不确定性及可视化分析

## 📁 项目结构

```
NewsGuard-AI/
├── backend/                 # 后端服务
│   ├── main.py             # FastAPI主程序
│   ├── model.py            # 模型定义(StudentNN, TextProcessor)
│   ├── cam_explainer.py    # Grad-CAM可视化模块
│   ├── config_file.py      # 配置参数
│   ├── requirements.txt    # Python依赖
│   ├── checkpoint/         # 模型权重(mse-sin_en, mse-sin_zh)
│   └── dataset/            # 数据文件(vocab, 词嵌入, 停用词)
├── components/             # React组件
│   ├── FileUpload.tsx     # 文件上传
│   ├── ResultDashboard.tsx # 结果展示面板
│   └── TextHighlighter.tsx # 文本注意力高亮
├── services/               # 前端服务
│   └── detectionService.ts # API调用
├── App.tsx                 # 主应用
├── types.ts                # TypeScript类型
└── package.json            # 前端配置
```

## 🛠️ 开发说明

**构建生产版本**:
```bash
npm run build  # 输出到 dist/
npm run preview
```

**API接口**: `POST /predict`

**请求参数**:
- `text` (string): 新闻文本内容
- `image` (file): 新闻配图

**响应字段**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `detected_language` | string | 语言 (zh/en) |
| `prediction` | string | 结果 (Real/Fake) |
| `confidence` | float | 置信度 (0-1) |
| `probabilities` | array | 概率分布 |
| `consistency.cosine_similarity` | float | 图文相似度 (-1~1) |
| `uncertainty.entropy` | float | 预测熵值 |
| `explainability.textAttention` | array | 文本注意力权重 |
| `explainability.imageCamOverlay` | string | Grad-CAM热力图Base64 |

## 📝 注意事项

1. **图片为必填项**：系统需要同时分析文本和图像
2. **模型加载时间**：首次启动或切换语言时约需10-30秒
3. **内存要求**：建议至少8GB RAM，GPU用户建议16GB以上
4. **浏览器兼容性**：推荐使用Chrome、Firefox、Edge等现代浏览器

## 📄 许可证

本项目为毕业设计项目，仅供学习和研究使用。