import React, { useMemo } from 'react';
import { TextAttentionItem } from '../types';

interface TextHighlighterProps {
  tokens: TextAttentionItem[];
  isFake: boolean;
  originalText?: string;
}

export const TextHighlighter: React.FC<TextHighlighterProps> = ({ tokens, isFake, originalText }) => {
  
  const normalizedWeights = useMemo(() => {
    if (tokens.length === 0) return [];
    
    const weights = tokens.map(t => t.weight);
    const minWeight = Math.min(...weights);
    const maxWeight = Math.max(...weights);
    
    if (maxWeight - minWeight < 1e-6) {
      return tokens.map(t => ({ ...t, weight: t.weight }));
    }
    
    return tokens.map(t => ({
      ...t,
      weight: (t.weight - minWeight) / (maxWeight - minWeight)
    }));
  }, [tokens]);

  if (!originalText) {
    return renderTokens(normalizedWeights, isFake);
  }

  const { elements, matchedCount } = useMemo(() => {
    const text = originalText;
    const elements: React.ReactNode[] = [];
    const usedTokens = new Set<number>();
    let matchedCount = 0;
    
    // 预处理：建立token索引映射（按长度降序排列，优先匹配长词）
    const tokensByLength: { token: string; lowerToken: string; index: number; weight: number }[] = [];
    normalizedWeights.forEach((t, idx) => {
      if (t.token && t.token.length > 0) {
        tokensByLength.push({
          token: t.token,
          lowerToken: t.token.toLowerCase(),
          index: idx,
          weight: t.weight
        });
      }
    });
    
    // 按长度降序排序
    tokensByLength.sort((a, b) => b.token.length - a.token.length);
    
    let i = 0;
    while (i < text.length) {
      let found = false;
      
      // 尝试匹配任意未使用的token
      for (const { token, lowerToken, index, weight } of tokensByLength) {
        if (usedTokens.has(index)) continue;
        
        // 检查当前位置是否匹配（不区分大小写）
        const textSub = text.substring(i).toLowerCase();
        if (textSub.startsWith(lowerToken)) {
          const style = getHighlightStyle(weight, isFake);
          usedTokens.add(index);
          matchedCount++;
          
          // 提取原始文本中的token（保留原始大小写和格式）
          const originalToken = text.substring(i, i + token.length);
          
          elements.push(
            <span
              key={`token-${index}-${i}`}
              className="relative inline-group"
              style={style}
            >
              {originalToken}
              {weight > 0.05 && (
                <span className="absolute top-full left-1/2 -translate-x-1/2 mt-1 hidden group-hover:block bg-slate-900/95 text-white text-[10px] px-2 py-1 rounded z-[100] font-mono whitespace-nowrap">
                  <span className={isFake ? 'text-orange-400' : 'text-teal-400'}>●</span> {(weight * 100).toFixed(1)}%
                </span>
              )}
            </span>
          );
          i += token.length;
          found = true;
          break;
        }
      }
      
      if (!found) {
        elements.push(<span key={`char-${i}`}>{text[i]}</span>);
        i++;
      }
    }
    
    return { elements, matchedCount };
  }, [originalText, tokens, normalizedWeights, isFake]);

  return (
    <div className="h-full flex flex-col">
       <div className="flex-1 p-6 overflow-y-auto max-h-[400px] text-sm leading-8 text-slate-700 text-justify tracking-wide">
         {elements.length > 0 ? (
           <span className="inline">{elements}</span>
         ) : (
           <div className="h-full flex items-center justify-center text-slate-400 italic">
              等待文本分析数据...
           </div>
         )}
       </div>

       <div className="px-4 py-2.5 border-t border-slate-100 bg-slate-50/80">
          <div className="flex justify-between items-center mb-2">
             <div className="flex items-center gap-3 text-[10px] text-slate-500 font-mono">
                <span>匹配: <span className="font-semibold text-slate-700">{matchedCount}/{tokens.length}</span></span>
                {tokens.length > 0 && (
                  <span>
                    Max: <span className={`font-semibold ${isFake ? 'text-orange-600' : 'text-teal-600'}`}>
                      {(Math.max(...normalizedWeights.map(t => t.weight)) * 100).toFixed(1)}%
                    </span>
                  </span>
                )}
             </div>
          </div>
          <div className="flex items-center gap-2">
             <span className="text-[9px] text-slate-400">低</span>
             <div className={`flex-1 h-2 rounded-full relative overflow-hidden ${
               isFake 
                 ? 'bg-gradient-to-r from-orange-200 via-orange-400 to-orange-600' 
                 : 'bg-gradient-to-r from-teal-200 via-teal-400 to-teal-600'
             }`}>
               {[0.25, 0.5, 0.75].map((pct) => (
                 <div
                   key={pct}
                   className="absolute top-0 bottom-0 w-px bg-white/50"
                   style={{ left: `${pct * 100}%` }}
                 />
               ))}
             </div>
             <span className="text-[9px] text-slate-400">高</span>
          </div>
       </div>
    </div>
  );
};

function getHighlightStyle(weight: number, isFake: boolean): React.CSSProperties {
  const baseColor = isFake
    ? { r: 251, g: 146, b: 60 }
    : { r: 20, g: 184, b: 166 };

  const opacity = 0.15 + 0.6 * weight;
  
  const style: React.CSSProperties = {
    backgroundColor: `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${opacity})`,
    padding: '1px 2px',
    borderRadius: '2px',
    transition: 'all 0.2s ease',
  };

  if (weight > 0.6) {
    style.boxShadow = `0 0 6px rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${0.4 * weight})`;
    style.fontWeight = '700';
    style.padding = '1px 3px';
  } else if (weight > 0.3) {
    style.borderBottom = `2px solid rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${0.6})`;
    style.fontWeight = '600';
  }

  return style;
}

function renderTokens(tokens: TextAttentionItem[], isFake: boolean) {
  return (
    <span className="inline text-justify tracking-wide">
      {tokens.map((item, index) => {
        const style = getHighlightStyle(item.weight, isFake);
        const isHighlighted = Object.keys(style).length > 0;

        return (
          <span
            key={index}
            className="relative inline-block group"
            style={isHighlighted ? style : {}}
          >
            {item.token}
            {item.weight > 0.05 && (
              <span className="absolute top-full left-1/2 -translate-x-1/2 mt-1 hidden group-hover:block bg-slate-900/95 text-white text-[10px] px-2 py-1 rounded z-[100] font-mono whitespace-nowrap">
                <span className={isFake ? 'text-orange-400' : 'text-teal-400'}>●</span> {(item.weight * 100).toFixed(1)}%
              </span>
            )}
          </span>
        );
      })}
    </span>
  );
}