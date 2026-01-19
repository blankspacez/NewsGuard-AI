import os
import io
import re
import pickle
import base64
import math
import numpy as np
from contextlib import asynccontextmanager
from typing import Dict, Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

# ==========================================
# 0. 路径与配置设置
# ==========================================

import config_file
from model import TextProcessor, StudentNN
from cam_explainer import GradCAMExplainer

# 2. 定义路径基准
# BACKEND_DIR: .../Project_Root/backend (main.py 所在目录)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT: .../Project_Root (backend 的上一级目录)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def process_config(raw_config):
    """处理列表格式的配置，转换为单值"""
    flat_config = {}
    for k, v in raw_config.items():
        if isinstance(v, list):
            flat_config[k] = v[0]
        else:
            flat_config[k] = v
    return flat_config


CONFIG = process_config(config_file.config)
print(f"Loaded Config: {CONFIG}")
print(f"Backend Dir: {BACKEND_DIR}")
print(f"Project Root: {PROJECT_ROOT}")

# 全局资源容器
RESOURCES = {
    'zh': {'model': None, 'processor': None},
    'en': {'model': None, 'processor': None}
}


def detect_language(text: str) -> str:
    if re.search(r'[\u4e00-\u9fff]', text):
        return 'zh'
    return 'en'

def load_resources(lang):
    """
    根据语言加载资源
    """
    vocab_path = os.path.join(BACKEND_DIR, 'dataset', f'vocab_{lang}.pkl')
    stop_path = os.path.join(BACKEND_DIR, 'dataset', f'stopwords_{lang}.txt')
    model_path = os.path.join(BACKEND_DIR, 'checkpoint', f'mse-sin_{lang}')
    w2v_path = os.path.join(BACKEND_DIR, 'dataset', f'train_{lang}.pkl')

    print(f"Loading {lang} resources from:")
    print(f"  Vocab: {vocab_path}")
    print(f"  StopWords: {stop_path}")
    print(f"  Model: {model_path}")
    print(f"  Word: {w2v_path}")

    processor = TextProcessor(lang, vocab_path, CONFIG['maxlen'], stop_path)
    CONFIG['embedding_weights'] = load_embedding_weights(w2v_path)

    # +1 是为了 Padding/Unknown，防止 index out of range
    vocab_size = len(processor.vocabulary) + 1
    model = StudentNN(CONFIG, vocab_size=vocab_size)

    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print("  Model weights loaded.")
        except Exception as e:
            print(f"  Error loading weights: {e}")
    else:
        print(f"  Warning: Checkpoint not found.")

    model.to(device)
    model.eval()
    return {'model': model, 'processor': processor}


def load_embedding_weights(pkl_path: str):
    """
    从pkl文件中加载embedding_weights

    参数:
        pkl_path: train.pkl文件的路径

    返回:
        embedding_weights: 词嵌入矩阵，形状为 (vocab_size + 1, w2v_dim)
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # train_zh.pkl 的结构: [X_train_tid, X_train, y_train, word_embeddings, relation]
    embedding_weights = data[3]
    print("embedding_weights.shape", embedding_weights.shape)

    return embedding_weights


def enable_mc_dropout(model: StudentNN):
    """
    将模型中所有 Dropout 层切换到训练模式，以便在推理阶段启用随机失活。
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def disable_mc_dropout(model: StudentNN):
    """恢复模型为 eval 状态（关闭 Dropout）。"""
    model.eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> Startup: Initializing models...")
    try:
        RESOURCES['zh'] = load_resources('zh')
        RESOURCES['en'] = load_resources('en')
    except Exception as e:
        print(f"Startup failed: {e}")
    yield
    print(">>> Shutdown: Cleaning up...")
    RESOURCES.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "online"}

img_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.post("/predict")
async def predict(
        text: str = Form(...),
        image: UploadFile = File(...)
):
    lang = detect_language(text)
    res = RESOURCES.get(lang)

    if not res or res['model'] is None:
        raise HTTPException(500, f"System not ready for language: {lang}")

    try:
        # Text
        text_tensor = res['processor'].transform(text).to(device)

        # Image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = img_pipeline(pil_image).unsqueeze(0).to(device)

        # Infer with features for consistency calculation
        model = res['model']
        with torch.no_grad():
            output, text_feature, image_feature = model._forward_internal(
                text_tensor, image_tensor, return_features=True
            )
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

        # Calculate consistency (cosine similarity between text and image features)
        consistency_data = None
        try:
            # text_feature and image_feature are both [B, 300]
            text_feat_norm = F.normalize(text_feature, p=2, dim=1)
            image_feat_norm = F.normalize(image_feature, p=2, dim=1)
            cosine_sim = (text_feat_norm * image_feat_norm).sum(dim=1)  # [B]
            consistency_data = {
                "cosine_similarity": float(cosine_sim[0].item())
            }
        except Exception as e:
            print(f"Consistency calculation failed: {e}")

        # Calculate uncertainty using MC Dropout
        uncertainty_data = None
        try:
            mc_iterations = 10
            enable_mc_dropout(model)
            mc_outputs = []
            with torch.no_grad():
                for _ in range(mc_iterations):
                    mc_out = model(text_tensor, image_tensor)
                    mc_probs = F.softmax(mc_out, dim=1)
                    mc_outputs.append(mc_probs)
            disable_mc_dropout(model)
            
            # Stack: [mc_iterations, B, num_classes]
            mc_stack = torch.stack(mc_outputs, dim=0)
            mean_probs = mc_stack.mean(dim=0)  # [B, num_classes]
            variance = mc_stack.var(dim=0).mean().item()  # scalar
            
            # Entropy of mean prediction
            entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=1).item()
            
            uncertainty_data = {
                "entropy": entropy,
                "variance": variance
            }
        except Exception as e:
            print(f"Uncertainty calculation failed: {e}")
            disable_mc_dropout(model)

        label_map = {0: "Real", 1: "Fake"}  # 根据训练时的Label调整

        # 文本“注意力”信息：从 TransformerBlock 中提取真实 self-attention 权重
        tokens = res['processor'].clean_and_tokenize(text)
        text_attention = []
        tokens = res['processor'].clean_and_tokenize(text)
        text_attention = []
        try:
            if tokens:
                # 1. 获取 Attention 矩阵
                # 假设 get_text_self_attention 返回的是最后一层的 attention
                attn_matrix = res['model'].get_text_self_attention(text_tensor)

                if attn_matrix is not None:
                    # attn_matrix shape: [Batch, Heads, L, L] 或 [Batch, L, L]
                    # 我们这里假设你已经处理过 Heads 的平均值，或者在这里处理
                    if len(attn_matrix.shape) == 4:
                        attn_matrix = attn_matrix.mean(dim=1)  # 对多头求平均 -> [B, L, L]

                    attn_matrix = attn_matrix[0]  # 取 Batch 第一个 -> [L, L]

                    L = attn_matrix.size(0)
                    t_len = min(len(tokens), L)

                    # 2. 截取有效区域 (假设是左侧 Padding，真实文本在最后)
                    attn_sub = attn_matrix[-t_len:, -t_len:]  # (t_len, t_len)

                    # 3. 【关键修改】Mask 掉对角线（移除自身注意力）
                    # 生成对角线 Mask
                    eye = torch.eye(t_len, device=attn_sub.device)
                    # 将对角线置为 0
                    attn_sub = attn_sub * (1 - eye)

                    # 4. 聚合分数
                    # dim=0 (列求和): 代表该 Token 被其他 Token 关注的程度 (Centrality/Importance)
                    token_scores = attn_sub.sum(dim=0)  # (t_len,)

                    # 5. 【改进】使用更鲁棒的归一化方法：百分位裁剪 + Softmax
                    # 先进行百分位裁剪，避免异常值影响
                    scores_np = token_scores.cpu().numpy()
                    p_low = np.percentile(scores_np, 5)
                    p_high = np.percentile(scores_np, 95)
                    scores_clipped = np.clip(scores_np, p_low, p_high)

                    # 归一化到合理范围，然后应用 softmax 增强对比度
                    if p_high - p_low > 1e-6:
                        scores_normalized = (scores_clipped - p_low) / (p_high - p_low + 1e-8)
                        # 应用 temperature scaling，拉开差距
                        temperature = 2.0  # 较小的 temperature 值会增强差异
                        scores_softmax = torch.softmax(torch.tensor(scores_normalized / temperature), dim=0)
                        token_scores = scores_softmax
                    else:
                        # 如果所有分数几乎相同，给予均匀分布
                        token_scores = torch.ones_like(token_scores) / len(token_scores)

                    # 6. 额外的平滑处理 (可选):
                    # 如果希望稍微平滑一点，不想让 0 分看起来完全透明，可以加一个基底
                    # token_scores = 0.2 + 0.8 * token_scores

                    used_tokens = tokens[-t_len:]
                    text_attention = [
                        {"token": tok, "weight": float(w.item())}
                        for tok, w in zip(used_tokens, token_scores)
                    ]
        except Exception as e:
            print(f"Text attention extraction failed: {e}")
            # 出错时的回退逻辑：均匀分布
            text_attention = [{"token": tok, "weight": 0.1} for tok in tokens]

        # 使用 Grad-CAM 生成图像热力图（基于最终分类决策的多模态梯度）
        cam_overlay_b64 = ""
        try:
            cam_explainer = GradCAMExplainer(res['model'])
            cam_image = cam_explainer.generate_cam(
                text_tensor,
                image_tensor,
                pil_image,
                target_class=int(pred.item()),
            )
            buf = io.BytesIO()
            cam_image.save(buf, format="PNG")
            cam_bytes = buf.getvalue()
            cam_overlay_b64 = "data:image/png;base64," + base64.b64encode(cam_bytes).decode("utf-8")
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")

        return {
            "detected_language": lang,
            "prediction": label_map.get(pred.item(), "Unknown"),
            "confidence": float(conf.item()),
            "probabilities": probs.cpu().numpy().tolist()[0],
            "consistency": consistency_data,
            "uncertainty": uncertainty_data,
            "explainability": {
                "textAttention": text_attention,
                "imageCamOverlay": cam_overlay_b64,
            },
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)