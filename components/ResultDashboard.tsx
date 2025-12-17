import React, { useState, useEffect } from 'react';
import { DetectionResult } from '../types';
import { TextHighlighter } from './TextHighlighter';
import { AlertTriangle, CheckCircle, Eye, EyeOff, Activity, Layers, ScanFace, GitCompare, HelpCircle } from 'lucide-react';

interface ResultDashboardProps {
  result: DetectionResult;
  originalImage: File | null;
}

export const ResultDashboard: React.FC<ResultDashboardProps> = ({ result, originalImage }) => {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [imgUrl, setImgUrl] = useState<string>("");

  useEffect(() => {
    if (originalImage) {
      const url = URL.createObjectURL(originalImage);
      setImgUrl(url);
      return () => URL.revokeObjectURL(url);
    }
  }, [originalImage]);

  const isFake = result.isFake;
  const isReal = !isFake;

  return (
    <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-700 ease-out">
      
      {/* 1. Top Verdict Panel */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col md:flex-row">
         
         {/* Verdict Status Color Bar */}
         <div className={`w-full md:w-2 ${isFake ? 'bg-rose-500' : 'bg-emerald-500'}`}></div>

         <div className="p-8 flex-1 flex flex-col md:flex-row items-center md:items-start gap-8">
            {/* Icon */}
            <div className={`p-4 rounded-full flex-shrink-0 ${isFake ? 'bg-rose-50' : 'bg-emerald-50'}`}>
               {isFake ? (
                 <AlertTriangle className="w-8 h-8 text-rose-500" />
               ) : (
                 <CheckCircle className="w-8 h-8 text-emerald-500" />
               )}
            </div>

            {/* Text Info */}
            <div className="flex-1 text-center md:text-left">
               <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">
                 Detection Verdict
               </div>
               <h2 className="text-3xl font-bold text-slate-900 mb-2">
                 {isFake ? '虚假新闻 (Fake News)' : '真实新闻 (Real News)'}
               </h2>
               <p className="text-sm text-slate-500 leading-relaxed max-w-lg">
                 {isFake 
                   ? '经过 ISMAF 模型综合分析，该内容在文本语义和图像特征上表现出高度的虚假特征，建议谨慎传播。' 
                   : '经过 ISMAF 模型综合分析，该内容特征与真实报道模式高度吻合，未发现明显的篡改或造假痕迹。'}
               </p>
            </div>

            {/* Score Visualization */}
            <div className="flex flex-col items-center justify-center p-4 bg-slate-50 rounded-xl border border-slate-100 min-w-[140px]">
               <div className="relative flex items-center justify-center mb-2">
                  <svg className="w-20 h-20 transform -rotate-90">
                     <circle cx="40" cy="40" r="36" stroke="currentColor" strokeWidth="6" fill="transparent" className="text-slate-200" />
                     <circle cx="40" cy="40" r="36" stroke="currentColor" strokeWidth="6" fill="transparent" 
                        strokeDasharray={36 * 2 * Math.PI}
                        strokeDashoffset={36 * 2 * Math.PI - (result.confidence / 100) * 36 * 2 * Math.PI}
                        className={isFake ? 'text-rose-500' : 'text-emerald-500'}
                     />
                  </svg>
                  <span className={`absolute text-xl font-bold font-mono ${isFake ? 'text-rose-600' : 'text-emerald-600'}`}>
                    {Math.round(result.confidence)}%
                  </span>
               </div>
               <span className="text-[10px] font-semibold text-slate-500 uppercase">置信度评分</span>
            </div>
         </div>
      </div>
      
      {/* 2. Metadata Bar */}
      <div className="flex gap-4 overflow-x-auto pb-2">
        <div className="px-4 py-2 bg-white rounded-lg border border-slate-200 text-xs font-mono text-slate-600 shadow-sm whitespace-nowrap">
          SOURCE: <span className="font-bold text-indigo-600">{result.model_used || 'AUTO'}</span>
        </div>
        <div className="px-4 py-2 bg-white rounded-lg border border-slate-200 text-xs font-mono text-slate-600 shadow-sm whitespace-nowrap">
          TIMESTAMP: <span className="text-slate-800">{new Date(result.timestamp).toLocaleTimeString()}</span>
        </div>
        <div className="px-4 py-2 bg-white rounded-lg border border-slate-200 text-xs font-mono text-slate-600 shadow-sm whitespace-nowrap">
          TOKENS: <span className="text-slate-800">{result.explainability.textAttention.length}</span>
        </div>
      </div>

      {/* 2.5 Consistency & Uncertainty Cards */}
      {(result.consistency || result.uncertainty) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Consistency Card */}
          {typeof result.consistency?.cosineSimilarity === 'number' && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 flex items-start gap-4">
              <div className="p-3 rounded-lg bg-blue-50 flex-shrink-0">
                <GitCompare className="w-6 h-6 text-blue-500" />
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-semibold text-slate-800">图文一致性</h4>
                  <span className="text-lg font-bold text-blue-600 font-mono">
                    {Math.round(((result.consistency.cosineSimilarity + 1) / 2) * 100)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 mb-2">
                  <div 
                    className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${((result.consistency.cosineSimilarity + 1) / 2) * 100}%` }}
                  />
                </div>
                <p className="text-xs text-slate-500">
                  文本特征与图像特征的余弦相似度，数值越高表示图文内容越匹配。
                </p>
              </div>
            </div>
          )}

          {/* Uncertainty Card */}
          {typeof result.uncertainty?.entropy === 'number' && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 flex items-start gap-4">
              <div className="p-3 rounded-lg bg-amber-50 flex-shrink-0">
                <HelpCircle className="w-6 h-6 text-amber-500" />
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-semibold text-slate-800">预测不确定性</h4>
                  <span className="text-lg font-bold text-amber-600 font-mono">
                    {result.uncertainty.entropy.toFixed(3)}
                  </span>
                </div>
                <div className="flex gap-4 text-xs text-slate-600 mb-2">
                  <span>熵值: <span className="font-mono font-semibold">{result.uncertainty.entropy.toFixed(4)}</span></span>
                  {typeof result.uncertainty.variance === 'number' && (
                    <span>方差: <span className="font-mono font-semibold">{result.uncertainty.variance.toFixed(4)}</span></span>
                  )}
                </div>
                <p className="text-xs text-slate-500">
                  基于 MC Dropout 的预测分布熵，数值越大表示模型对该样本的判断越不确定。
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 3. Unified CAM Card: Text + Image */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="bg-slate-900 text-white text-xs font-bold px-4 py-2 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-2">
              <Layers size={14} /> 文本语义注意力权重分布
            </span>
            <span className="text-slate-500">/</span>
            <span className="flex items-center gap-2">
              <ScanFace size={14} /> 视觉梯度加权类激活图
            </span>
          </div>
          {result.explainability.imageCamOverlay ? (
            <button
              onClick={() => setShowHeatmap(!showHeatmap)}
              className={`text-[10px] flex items-center gap-1 px-2 py-1 rounded transition-colors ${
                showHeatmap 
                  ? 'bg-emerald-500 text-white hover:bg-emerald-600' 
                  : 'border border-slate-500 hover:bg-slate-800'
              }`}
            >
              {showHeatmap ? (
                <><EyeOff size={10} /> 隐藏热力图</>
              ) : (
                <><Eye size={10} /> 显示热力图</>
              )}
            </button>
          ) : (
            <span className="text-[10px] text-slate-500">热力图不可用</span>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-0 h-full">
          {/* 左侧：文本注意力 */}
          <div className="border-r border-slate-200 max-h-[420px]">
            <TextHighlighter tokens={result.explainability.textAttention} isFake={isFake} />
          </div>

          {/* 右侧：图像 + Grad-CAM */}
          <div className="p-4 flex flex-col items-center justify-center bg-slate-50 h-full">
            {imgUrl ? (
              <div className="relative w-full max-w-[520px] rounded-lg overflow-hidden border border-slate-200 shadow-sm bg-slate-900">
                <div className="relative w-full" style={{ paddingTop: '60%' }}>
                  <img
                    src={imgUrl}
                    className="absolute inset-0 w-full h-full object-contain z-0"
                    alt="Original"
                  />
                  {result.explainability.imageCamOverlay && (
                    <img
                      src={result.explainability.imageCamOverlay}
                      className={`absolute inset-0 w-full h-full object-contain z-10 transition-opacity duration-500 mix-blend-screen ${
                        showHeatmap ? 'opacity-100' : 'opacity-0'
                      }`}
                      alt="Heatmap"
                    />
                  )}
                </div>
                <div className="flex justify-between items-center px-3 py-2 bg-slate-900/90">
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${showHeatmap && result.explainability.imageCamOverlay ? 'bg-emerald-600 text-white' : 'bg-slate-800 text-slate-300'}`}>
                    {showHeatmap && result.explainability.imageCamOverlay ? 'Grad-CAM ON' : 'Original'}
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">
                    ResNet50
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <Activity className="w-12 h-12 text-slate-200 mx-auto mb-3" />
                <p className="text-sm text-slate-400">无图像输入</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};