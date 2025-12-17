import React, { useRef, useState } from 'react';
import { Upload, X, Image as ImageIcon, FileImage } from 'lucide-react';

interface FileUploadProps {
  onFileSelect: (file: File | null) => void;
  selectedFile: File | null;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, selectedFile }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
      const objectUrl = URL.createObjectURL(file);
      setPreview(objectUrl);
    }
  };

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    onFileSelect(null);
    setPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full group">
      <input
        type="file"
        accept="image/*"
        className="hidden"
        ref={fileInputRef}
        onChange={handleFileChange}
      />
      
      {!selectedFile ? (
        <div 
          onClick={() => fileInputRef.current?.click()}
          className="border border-dashed border-slate-300 rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer hover:border-indigo-500 hover:bg-indigo-50/30 transition-all duration-300 h-32 bg-slate-50/50"
        >
          <Upload className="w-5 h-5 text-slate-400 mb-2 group-hover:text-indigo-500 transition-colors" />
          <p className="text-xs font-medium text-slate-600">点击上传图像 <span className="text-rose-500">*</span></p>
          <p className="text-[10px] text-slate-400 mt-1 font-mono">JPG / PNG (必填)</p>
        </div>
      ) : (
        <div className="relative rounded-lg overflow-hidden border border-slate-200 bg-white p-2 flex items-center gap-3 shadow-sm">
          <div className="w-12 h-12 rounded bg-slate-100 overflow-hidden flex-shrink-0 border border-slate-100 relative">
             {preview ? (
                <img src={preview} alt="Preview" className="w-full h-full object-cover" />
              ) : (
                <FileImage className="w-6 h-6 text-slate-300 m-auto mt-3" />
              )}
          </div>
          
          <div className="flex-1 min-w-0">
             <p className="text-xs font-semibold text-slate-700 truncate font-mono">{selectedFile.name}</p>
             <p className="text-[10px] text-slate-400 mt-0.5">{(selectedFile.size / 1024).toFixed(1)} KB</p>
          </div>

          <button 
            onClick={handleClear}
            className="p-1.5 rounded-md hover:bg-slate-100 text-slate-400 hover:text-red-500 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
};