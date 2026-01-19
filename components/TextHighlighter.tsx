import React from 'react';
import { TextAttentionItem } from '../types';

interface TextHighlighterProps {
  tokens: TextAttentionItem[];
  isFake: boolean;
}

export const TextHighlighter: React.FC<TextHighlighterProps> = ({ tokens, isFake }) => {
  // 增强版本：改进权重对比度和视觉编码

  const getStyle = (weight: number) => {
    // 过滤掉极低权重的 token，减少视觉干扰
    if (weight <= 0.01) return {};

    // 使用 JET 风格的渐变色谱方案
    // 低权重：冷色调（蓝/紫）
    // 中权重：过渡色（绿/黄）
    // 高权重：暖色调（红/橙）

    let baseColor: { r: number; g: number; b: number };

    if (weight < 0.2) {
      // 冷色调区间（蓝-紫）
      const t = weight / 0.2;
      baseColor = {
        r: Math.floor(30 + t * 20),
        g: Math.floor(64 + t * 20),
        b: Math.floor(175 + t * 30),
      };
    } else if (weight < 0.5) {
      // 过渡色区间（绿-黄）
      const t = (weight - 0.2) / 0.3;
      baseColor = {
        r: Math.floor(50 + t * 180),
        g: Math.floor(84 + t * 120),
        b: Math.floor(205 - t * 180),
      };
    } else {
      // 暖色调区间（橙-红）
      const t = (weight - 0.5) / 0.5;
      baseColor = {
        r: Math.floor(230 + t * 25),
        g: Math.floor(204 - t * 164),
        b: Math.floor(25 - t * 20),
      };
    }

    // 动态透明度：高权重更不透明
    const alpha = 0.15 + 0.7 * weight;

    // 构建样式对象
    const style: React.CSSProperties = {
      backgroundColor: `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${alpha})`,
      padding: '1px 3px',
      margin: '0 1px',
      borderRadius: '3px',
      display: 'inline-block',
      transition: 'all 0.2s ease',
    };

    // 高权重 token 额外的视觉强调
    if (weight > 0.6) {
      style.boxShadow = `0 0 10px rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${0.5 + 0.3 * weight})`;
      style.fontWeight = '700';
      style.textShadow = `0 0 4px rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, 0.5)`;
      style.padding = '1px 4px';
    } else if (weight > 0.3) {
      // 中权重添加下划线
      style.borderBottom = `2px solid rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${0.5 + 0.4 * weight})`;
      style.fontWeight = '500';
    }

    return style;
  };

  return (
    <div className="h-full flex flex-col">
       <div className="flex-1 p-6 overflow-y-auto max-h-[400px] text-sm leading-8 font-sans text-slate-700 text-justify">
        {tokens.length > 0 ? (
          tokens.map((item, index) => {
            const style = getStyle(item.weight);
            const isHighlighted = Object.keys(style).length > 0;

            return (
              <span
                key={index}
                className={`inline-block relative group cursor-default ${
                  isHighlighted ? 'transform hover:scale-105 hover:z-10' : ''
                }`}
                style={isHighlighted ? style : { padding: '0.5px 2px', margin: '0 0.5px' }}
              >
                {item.token}
                {item.weight > 0.05 && (
                   <span className="absolute top-full left-1/2 -translate-x-1/2 mt-1 hidden group-hover:block bg-slate-900/95 text-white text-[10px] px-2.5 py-1 rounded-lg whitespace-nowrap z-[100] font-mono shadow-lg pointer-events-none backdrop-blur-sm border border-slate-700/50">
                      <div className="flex items-center gap-1.5">
                        <span className={isFake ? 'text-rose-400' : 'text-emerald-400'}>●</span>
                        <span className="font-semibold">{(item.weight * 100).toFixed(1)}%</span>
                        <span className="text-slate-400">attention</span>
                      </div>
                   </span>
                )}
              </span>
            );
          })
        ) : (
          <div className="h-full flex items-center justify-center text-slate-400 italic">
             等待文本分析数据...
          </div>
        )}
       </div>

       {/* 增强的图例区域 */}
       <div className="px-4 py-2.5 border-t border-slate-100 bg-slate-50/80">
          <div className="flex justify-between items-center mb-2">
             <div className="flex items-center gap-3 text-[10px] text-slate-500">
                <span className="font-mono">Tokens: <span className="font-semibold text-slate-700">{tokens.length}</span></span>
                {tokens.length > 0 && (
                  <span className="font-mono">
                    Max: <span className={`font-semibold ${isFake ? 'text-rose-600' : 'text-emerald-600'}`}>
                      {(Math.max(...tokens.map(t => t.weight)) * 100).toFixed(1)}%
                    </span>
                  </span>
                )}
             </div>
          </div>
          <div className="flex items-center gap-2">
             <span className="text-[9px] text-slate-400 font-medium">Low</span>
             <div className={`flex-1 h-2 rounded-full relative overflow-hidden ${
               isFake ? 'bg-gradient-to-r from-rose-100 via-rose-300 to-rose-500' : 'bg-gradient-to-r from-emerald-100 via-emerald-300 to-emerald-500'
             }`}>
               {/* 刻度标记 */}
               {[0.25, 0.5, 0.75].map((pct) => (
                 <div
                   key={pct}
                   className="absolute top-0 bottom-0 w-px bg-white/50"
                   style={{ left: `${pct * 100}%` }}
                 />
               ))}
             </div>
             <span className="text-[9px] text-slate-400 font-medium">High</span>
          </div>
       </div>
    </div>
  );
};