 export interface TextAttentionItem {
   token: string;
   weight: number; // 0 to 1
 }
 
 export interface DetectionResult {
   isFake: boolean;
   confidence: number; // 0 to 100
   verdict: string;
   model_used?: string;
   timestamp: string;
   originalText: string; // 原始用户输入文本
   // 图文一致性与不确定性信息
   consistency?: {
     // 文本特征与图像特征的余弦相似度，范围约为 [-1, 1]
     cosineSimilarity: number;
   };
   uncertainty?: {
     // MC Dropout 预测分布的熵，数值越大代表模型越不确定
     entropy: number;
     // 各类别概率的平均方差，反映整体预测波动
     variance: number;
   };
   explainability: {
     textAttention: TextAttentionItem[];
     imageCamOverlay: string; // Base64 string
   };
 }

export interface AnalysisRequest {
  text: string;
  image: File | null; // Required: 图片是必填项
  // Model selection removed, handled dynamically by backend
}
