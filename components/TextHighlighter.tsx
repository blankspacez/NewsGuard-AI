import React from 'react';
import { TextAttentionItem } from '../types';

interface TextHighlighterProps {
  tokens: TextAttentionItem[];
  isFake: boolean;
}

export const TextHighlighter: React.FC<TextHighlighterProps> = ({ tokens, isFake }) => {
  // Use a monochromatic scale for attention to look more "scientific"
  // Fake: Red scale, Real: Green scale, or Neutral Blue scale.
  // Let's use the verdict color but keep it subtle.
  
  const getStyle = (weight: number) => {
    // 权重区间为 [0,1]，我们用一个非线性映射来拉大高权重与中低权重的差距
    // 低权重基本透明，高权重接近不透明
    if (weight <= 0.02) return {};

    // 提高高权重的对比度：指数增强 + 最小可见值
    const boosted = Math.pow(weight, 0.7); // 0.4 -> ~0.51, 1.0 -> 1.0
    const alpha = 0.15 + 0.85 * boosted;   // 映射到 [0.15, 1.0]
    
    if (isFake) {
       return {
          backgroundColor: `rgba(244, 63, 94, ${alpha})`, // Rose
          // 仅对最高注意力词加下划线，进一步强调
          borderBottom: weight > 0.75 ? '2px solid rgba(244, 63, 94, 1)' : 'none'
       };
    } else {
       return {
          backgroundColor: `rgba(16, 185, 129, ${alpha})`, // Emerald
          borderBottom: weight > 0.75 ? '2px solid rgba(16, 185, 129, 1)' : 'none'
       };
    }
  };

  return (
    <div className="h-full flex flex-col">
       <div className="flex-1 p-6 overflow-y-auto max-h-[400px] text-sm leading-8 font-sans text-slate-700 text-justify">
        {tokens.length > 0 ? (
          tokens.map((item, index) => (
            <span
              key={index}
              className="inline-block px-0.5 mx-[1px] rounded-sm transition-all hover:bg-slate-200 relative group cursor-default"
              style={getStyle(item.weight)}
            >
              {item.token}
              {item.weight > 0.1 && (
                 <span className="absolute top-full left-1/2 -translate-x-1/2 mt-1 hidden group-hover:block bg-slate-800 text-white text-[10px] px-2 py-1 rounded whitespace-nowrap z-[100] font-mono shadow-lg pointer-events-none">
                    Attn: {(item.weight * 100).toFixed(2)}%
                 </span>
              )}
            </span>
          ))
        ) : (
          <div className="h-full flex items-center justify-center text-slate-400 italic">
             等待文本分析数据...
          </div>
        )}
       </div>
       
       <div className="px-4 py-2.5 border-t border-slate-100 bg-slate-50/80 flex justify-between items-center text-[10px] text-slate-500">
          <span className="font-mono">Tokens: <span className="font-semibold text-slate-700">{tokens.length}</span></span>
          <div className="flex items-center gap-1.5">
             <span className="text-slate-400">Low</span>
             <div className={`w-16 h-1.5 rounded-full ${isFake ? 'bg-gradient-to-r from-rose-100 to-rose-500' : 'bg-gradient-to-r from-emerald-100 to-emerald-500'}`}></div>
             <span className="text-slate-400">High</span>
          </div>
       </div>
    </div>
  );
};