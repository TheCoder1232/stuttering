import React from "react";
import { useDropzone } from "react-dropzone";
import { Mic, Square, Loader2, UploadCloud } from "lucide-react";

const RecordingZone = ({ 
  recording, 
  analyzing, 
  results, 
  onStartRecording, 
  onStopRecording, 
  onDrop 
}) => {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'audio/*': [] },
    disabled: recording || analyzing || results
  });

  if (results) return null;

  return (
    <div 
      {...getRootProps()} 
      className={`relative rounded-2xl border-2 border-dashed transition-all duration-300 min-h-[300px] flex flex-col items-center justify-center p-8 cursor-pointer ${
        isDragActive ? "border-indigo-500 bg-indigo-50" : "border-slate-300 bg-white hover:border-indigo-300"
      } ${analyzing ? "pointer-events-none opacity-80" : ""}`}
    >
      <input {...getInputProps()} />
      {analyzing ? (
        <div className="flex flex-col items-center text-indigo-600 animate-in fade-in">
          <Loader2 className="w-16 h-16 animate-spin mb-4" />
          <h3 className="text-xl font-bold">Analyzing Audio...</h3>
        </div>
      ) : (
        <>
          <button
            onClick={recording ? onStopRecording : onStartRecording}
            className={`z-20 w-24 h-24 rounded-full flex items-center justify-center transition-all shadow-xl mb-6 ${
              recording ? "bg-red-500 hover:bg-red-600 animate-pulse" : "bg-indigo-600 hover:bg-indigo-700 hover:scale-105"
            }`}
          >
            {recording ? <Square className="w-10 h-10 text-white fill-current" /> : <Mic className="w-10 h-10 text-white" />}
          </button>
          <div className="text-center space-y-2">
            <h3 className="text-lg font-bold text-slate-700">{recording ? "Recording..." : "Record Speech"}</h3>
            {!recording && (
              <div className="flex items-center gap-2 text-slate-500 bg-slate-100 px-4 py-2 rounded-full mx-auto w-fit">
                <UploadCloud className="w-4 h-4" />
                <span className="text-sm">or Drop audio file here</span>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default RecordingZone;
