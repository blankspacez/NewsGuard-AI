import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Optional

class GradCAMExplainer:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device 
        self.feature_maps = None
        self.gradients = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        try:
            # 确保层级正确，ResNet 通常是 layer4
            target_layer = self.model.image_embedding.model.layer4
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
            print("Grad-CAM Hook registered.")
        except AttributeError as e:
            print(f"Error registering hook: {e}")

    def generate_cam(self,
                      text_tensor: torch.Tensor,
                      image_tensor: torch.Tensor,
                      original_image: Image.Image,
                      target_class: Optional[int] = None) -> Image.Image:

        # 重置 hooks 捕获的数据，确保每次都是干净的状态
        self.feature_maps = None
        self.gradients = None

        # 1. 格式标准化
        original_image = original_image.convert('RGB')
        img_np = np.array(original_image)
        h, w = img_np.shape[:2]

        self.model.eval()
        self.model.zero_grad()
        
        text_tensor = text_tensor.to(self.device)
        image_tensor = image_tensor.to(self.device)
        
        # 2. 前向与反向传播
        with torch.enable_grad():
            logits = self.model(text_tensor, image_tensor)
            if target_class is None:
                target_class = logits.argmax(dim=1).item()
            
            # score = logits[:, target_class].sum() # 这种写法有时梯度会断
            score = logits[0, target_class] # 更安全的写法
            
            self.model.zero_grad()
            score.backward()
        
        if self.gradients is None or self.feature_maps is None:
            return original_image

        # 3. Grad-CAM 核心计算
        gradients = self.gradients[0]
        feature_maps = self.feature_maps[0]
        
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * feature_maps, dim=0, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        # 4. [修复核心 1] 鲁棒的归一化与 Gamma 校正
        # 避免除以 0
        if np.max(cam) == 0:
            return original_image
            
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-9) # 归一化到 0-1
        
        # [关键修改] Gamma 校正：拉伸中间色调 (Green/Yellow)
        # 原始数据往往两极分化。0.6 的幂次可以把 0.2 提升到 0.38，把 0.5 提升到 0.66
        # 这会让原本只有“蓝红”的图，出现更多的“绿色和黄色”过渡
        cam = np.power(cam, 0.6) 

        # 5. [修复核心 2] Resize + 强高斯模糊
        heatmap = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # 动态计算模糊核大小，保证大图小图效果一致
        # 增大模糊半径，消除“斑块感”
        blur_k = int(min(w, h) * 0.1) # 比如 224的图用 21，1000的图用 99
        if blur_k % 2 == 0: blur_k += 1
        if blur_k < 3: blur_k = 3
            
        heatmap = cv2.GaussianBlur(heatmap, (blur_k, blur_k), 0)
        
        # [关键修改] 不要再做第二次 /np.max() 归一化了！
        # 模糊会自然降低峰值（让红色中心变柔和），如果再次强行拉回 1.0，斑块就回来了。
        # 只需要简单的截断即可
        heatmap = np.clip(heatmap, 0, 1)
            
        # 6. 应用 JET 色谱
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # 7. 混合
        # 提高一点热力图的权重 (0.4 -> 0.5) 让颜色更鲜艳，对抗 Gamma 带来的变淡
        overlay = cv2.addWeighted(img_np, 0.5, heatmap_colored, 0.5, 0)

        return Image.fromarray(overlay)