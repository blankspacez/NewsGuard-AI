import React, { useState, useEffect } from 'react';
import { LayoutDashboard, ShieldCheck, Loader2, Sparkles, BrainCircuit, Info, ChevronRight, Binary } from 'lucide-react';
import { FileUpload } from './components/FileUpload';
import { ResultDashboard } from './components/ResultDashboard';
import { DetectionResult } from './types';
import { detectFakeNews, checkBackendHealth } from './services/detectionService';

const App: React.FC = () => {
  const [textInput, setTextInput] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);

  // Check backend health status periodically
  useEffect(() => {
    const checkHealth = async () => {
      const isOnline = await checkBackendHealth();
      setBackendOnline(isOnline);
    };

    // Initial check
    checkHealth();

    // Set up interval to check every 5 seconds
    const interval = setInterval(checkHealth, 5000);

    // Clean up interval on component unmount
    return () => clearInterval(interval);
  }, []);

  const handleAnalyze = async () => {
    if (!textInput.trim()) {
      alert("请输入需要检测的新闻文本。");
      return;
    }

    if (!selectedFile) {
      alert("请上传新闻相关图片。图片是必填项。");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const data = await detectFakeNews({
        text: textInput,
        image: selectedFile
      });
      setResult(data);
    } catch (error) {
      console.error("Detection failed", error);
      alert("分析失败，请确保后端服务已启动。");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen text-slate-900 pb-20 font-sans selection:bg-indigo-100 selection:text-indigo-900">
      
      {/* Background Ambience */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
         <div className="absolute top-0 left-0 w-full h-[500px] bg-gradient-to-b from-white via-slate-50 to-transparent opacity-80" />
      </div>

      {/* Header - Academic Style */}
      <header className="sticky top-0 z-50 bg-white/70 backdrop-blur-xl border-b border-slate-200 shadow-[0_2px_10px_rgba(0,0,0,0.02)]">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="bg-slate-900 p-2 rounded-lg shadow-lg shadow-slate-900/20">
              <Binary className="w-5 h-5 text-white" />
            </div>
            <div className="flex flex-col">
              <h1 className="font-bold text-lg tracking-tight text-slate-900 leading-tight">
                Fake News Detection <span className="font-light text-slate-500">System</span>
              </h1>
              <span className="text-[10px] font-medium tracking-widest text-slate-500 uppercase">
                多模态虚假新闻检测系统
              </span>
            </div>
          </div>
          
          <div className="hidden md:flex items-center gap-6">
            <div className={`flex items-center gap-2 text-xs font-medium px-3 py-1 rounded-md border ${
              backendOnline 
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200' 
                : 'bg-rose-50 text-rose-700 border-rose-200'
            }`}>
               <span className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-500' : 'bg-rose-500'}`}></span>
               {backendOnline ? 'System Online' : 'System Offline'}
            </div>
            <div className="h-4 w-[1px] bg-slate-300"></div>
            <div className="flex items-center gap-2 text-xs text-slate-600">
               <BrainCircuit size={14} className="text-indigo-600" /> 
               <span className="font-mono">Model: ISMAF-Student</span>
            </div>
          </div>
        </div>
      </header>

      <main className="relative z-10 max-w-7xl mx-auto px-6 py-12">
        
        {/* Title Section */}
        <div className="mb-10">
           <h2 className="text-3xl font-bold text-slate-900 tracking-tight mb-2">检测控制台</h2>
           <p className="text-slate-500 text-sm max-w-2xl leading-relaxed">
             请输入待检测的新闻文本及图片（图片为必填项）。系统将利用深度神经网络 (ISMAF) 提取跨模态特征，自动判定内容的真实性并提供可解释性热力图。
           </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Input Panel (4 cols) */}
          <div className="lg:col-span-4 flex flex-col gap-6">
            
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="bg-slate-50/50 px-5 py-3 border-b border-slate-100 flex justify-between items-center">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                  <LayoutDashboard size={14} /> 数据输入
                </span>
                <span className="text-[10px] text-slate-400 font-mono">INPUT_STREAM</span>
              </div>
              
              <div className="p-5 flex flex-col gap-5">
                <div className="space-y-2">
                  <label className="text-xs font-semibold text-slate-700 flex justify-between">
                    文本内容
                  </label>
                  <textarea
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder="在此输入新闻正文..."
                    className="w-full h-48 p-4 rounded-lg border border-slate-200 bg-slate-50 text-sm leading-relaxed placeholder:text-slate-400 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 focus:bg-white outline-none resize-none transition-all duration-200 font-mono text-xs"
                    disabled={!backendOnline}
                  />
                  <div className="flex justify-end">
                     <span className="text-[10px] text-slate-400">{textInput.length} chars</span>
                  </div>
                </div>

                <div className="space-y-2">
                   <div className="flex justify-between items-end">
                      <label className="text-xs font-semibold text-slate-700">图像附件</label>
                      <span className="text-[10px] text-rose-500 bg-rose-50 px-1.5 rounded font-semibold">必填</span>
                   </div>
                   <FileUpload onFileSelect={setSelectedFile} selectedFile={selectedFile} disabled={!backendOnline} />
                </div>
              </div>
              
              <div className="p-5 bg-slate-50/30 border-t border-slate-100">
                <button
                  onClick={handleAnalyze}
                  disabled={loading || !textInput || !selectedFile || !backendOnline}
                  className={`w-full py-3 rounded-lg font-medium text-sm flex items-center justify-center gap-2 transition-all duration-200 
                    ${loading || !textInput || !selectedFile || !backendOnline
                      ? 'bg-slate-200 text-slate-400 cursor-not-allowed' 
                      : 'bg-slate-900 text-white hover:bg-slate-800 hover:shadow-lg hover:shadow-slate-900/20 active:translate-y-0.5'
                    }`}
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>正在推理...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>执行分析</span>
                    </>
                  )}
                </button>
                
                {!backendOnline && (
                  <div className="mt-3 text-xs text-rose-600 text-center">
                    ⚠️ 后端服务未启动，请先启动后端服务再进行分析
                  </div>
                )}
              </div>
            </div>
            
            {/* System Status Card */}
            <div className="bg-white/60 rounded-lg border border-slate-200 p-4 backdrop-blur-sm">
               <div className="flex items-start gap-3">
                  <Info className="w-4 h-4 text-indigo-500 mt-0.5 flex-shrink-0" />
                  <div className="text-xs text-slate-600 leading-relaxed">
                    <p className="font-semibold text-slate-800 mb-1">自适应模型路由</p>
                    系统集成了 PHEME (英文) 与 WEIBO (中文) 两个数据集的预训练权重。推理引擎将根据输入文本的语言特征自动加载对应的计算图。
                  </div>
               </div>
            </div>

          </div>

          {/* Right Column: Results Panel (8 cols) */}
          <div className="lg:col-span-8">
            {result ? (
              <ResultDashboard result={result} originalImage={selectedFile} />
            ) : (
              <div className="h-full min-h-[500px] bg-white rounded-xl border border-slate-200 border-dashed flex flex-col items-center justify-center text-slate-400 relative overflow-hidden group">
                 {/* Subtle Grid Background inside empty state */}
                 <div className="absolute inset-0 bg-[linear-gradient(to_right,#f1f5f9_1px,transparent_1px),linear-gradient(to_bottom,#f1f5f9_1px,transparent_1px)] bg-[size:24px_24px] opacity-50"></div>
                 
                 <div className="relative z-10 flex flex-col items-center p-8 transition-transform duration-500 group-hover:scale-105">
                    <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mb-6 border border-slate-100 shadow-sm">
                      <ShieldCheck className="w-10 h-10 text-slate-300" />
                    </div>
                    <h3 className="text-lg font-semibold text-slate-700 mb-2">等待数据输入</h3>
                    <p className="text-sm text-slate-400 text-center max-w-sm">
                      请在左侧面板完成数据加载。分析结果包括真伪判定置信度、文本注意力分布及图像热力图。
                    </p>
                 </div>
              </div>
            )}
          </div>

        </div>
      </main>
      
      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-8 mt-12">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center text-xs text-slate-400">
           <div className="flex items-center gap-4">
              <span className="font-mono">Build v2.1.0</span>
              <span>•</span>
              <span>Based on ISMAF Architecture</span>
           </div>
        </div>
      </footer>
    </div>
  );
};

export default App;