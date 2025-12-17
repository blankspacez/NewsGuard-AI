import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Optional

class GradCAMExplainer:
    """
    针对 StudentNN 多模态模型的 Grad-CAM 实现
    """
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device # 自动获取模型所在设备
        self.feature_maps = None
        self.gradients = None
        
        # 注册 Hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """
        注册 Hook 到 ResNet50 的最后一个卷积层 (layer4)
        """
        def forward_hook(module, input, output):
            self.feature_maps = output
        
        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] 是对应 feature_maps 的梯度
            self.gradients = grad_output[0]
        
        # 定位到 StudentNN -> image_embedding (ResNet50_Encoder) -> model (ResNet50) -> layer4
        # 请根据你实际的模型层级名称确认是否需要调整
        try:
            target_layer = self.model.image_embedding.model.layer4
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
            print("Grad-CAM Hook registered on ResNet layer4.")
        except AttributeError as e:
            print(f"Error registering hook: {e}")
            print("请检查模型结构属性名是否为 .image_embedding.model.layer4")

    def generate_cam(self,
                     text_tensor: torch.Tensor,
                     image_tensor: torch.Tensor,
                     original_image: Image.Image,
                     target_class: Optional[int] = None) -> Image.Image:
        """
        生成 Grad-CAM 热力图叠加图
        """
        # 1. 准备模型和数据
        self.model.eval()
        # 确保梯度清零
        self.model.zero_grad()
        
        text_tensor = text_tensor.to(self.device)
        image_tensor = image_tensor.to(self.device)
        
        # 关键：即便是在 eval 模式下，也需要允许梯度计算
        with torch.enable_grad():
            # 2. 前向传播 (完整的多模态模型)
            logits = self.model(text_tensor, image_tensor)
            
            # 3. 确定目标类别
            if target_class is None:
                # 默认解释预测概率最大的那个类别
                target_class = logits.argmax(dim=1).item()
            
            # 4. 反向传播
            # 获取目标类别的 logit 分数
            score = logits[:, target_class].sum()
            
            # 这是一个关键步骤：清空之前的梯度，然后反向传播
            self.model.zero_grad()
            score.backward()
        
        # 5. 生成 CAM（LayerCAM 风格：逐像素梯度 * 特征图，更细粒度）
        gradients = self.gradients
        feature_maps = self.feature_maps
        
        if gradients is None or feature_maps is None:
            print("Warning: No gradients or feature maps captured.")
            return original_image

        # --- LayerCAM 核心逻辑 ---
        # 1. 对梯度做 ReLU，只保留正向贡献（正梯度）
        # 2. 与特征图做逐元素乘法 (Element-wise multiplication)，而不是像 Grad-CAM 那样先求均值
        target = feature_maps * F.relu(gradients)

        # 3. 在 Channel 维度求和
        cam = target.sum(dim=1, keepdim=True)  # [1, 1, 7, 7]
        cam = F.relu(cam)  # 去除负值
        cam = cam.squeeze().detach().cpu().numpy()

        # 4. 归一化与缩放
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        cam_resized = cv2.resize(cam, (original_image.width, original_image.height))

        return self._apply_colormap(cam_resized, original_image)
    
    def _apply_colormap(self, cam: np.ndarray, 
                        original_image: Image.Image,
                        alpha: float = 0.5) -> Image.Image:
        """将 CAM 热力图叠加到原图"""
        # 转换为 0-255 的 uint8
        cam_uint8 = np.uint8(255 * cam)
        
        # 应用热力图配色 (JET 是最常用的彩虹色)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        
        # OpenCV 默认是 BGR，转回 RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 调整原图尺寸以防万一
        original_np = np.array(original_image)
        if original_np.shape[:2] != heatmap.shape[:2]:
            original_np = np.array(original_image.resize((heatmap.shape[1], heatmap.shape[0])))
        
        # 图像融合
        blended = cv2.addWeighted(original_np, 1 - alpha, heatmap, alpha, 0)
        
        return Image.fromarray(blended)