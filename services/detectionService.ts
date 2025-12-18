import { AnalysisRequest, DetectionResult } from '../types';

// Assuming backend runs on port 8000 locally
const API_BASE_URL = 'http://localhost:8000';

export const detectFakeNews = async (request: AnalysisRequest): Promise<DetectionResult> => {
  if (!request.image) {
    throw new Error('图片是必填项，请上传新闻相关图片。');
  }

  const formData = new FormData();
  formData.append('text', request.text);
  formData.append('image', request.image);

  const API_URL = `${API_BASE_URL}/predict`;

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    
    // Map backend response to frontend DetectionResult format
    const isFake = data.prediction === "Fake";
    const confidencePercent = (data.confidence * 100); // MC Dropout 平均置信度 -> 百分比
    
    return {
      isFake,
      confidence: confidencePercent,
      verdict: isFake ? "虚假新闻" : "真实新闻",
      model_used: data.detected_language === "zh" ? "WEIBO" : "PHEME",
      timestamp: new Date().toISOString(),
      consistency: data.consistency
        ? {
            cosineSimilarity: data.consistency.cosine_similarity ?? 0,
          }
        : undefined,
      uncertainty: data.uncertainty
        ? {
            entropy: data.uncertainty.entropy ?? 0,
            variance: data.uncertainty.variance ?? 0,
          }
        : undefined,
      explainability: {
        textAttention: data.explainability?.textAttention || [],
        imageCamOverlay: data.explainability?.imageCamOverlay || "",
      },
    };
  } catch (error) {
    console.error("Detection API call failed", error);
    // Propagate error to UI
    throw error;
  }
};

export const checkBackendHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      return false;
    }
    
    const data = await response.json();
    return data.status === 'online';
  } catch (error) {
    console.error("Health check failed", error);
    return false;
  }
};