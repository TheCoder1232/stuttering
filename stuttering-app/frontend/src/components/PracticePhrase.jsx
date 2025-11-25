import React from "react";
import { ChevronRight, ChevronLeft } from "lucide-react";

const PracticePhrase = ({ phrases, currentIndex, onPrevious, onNext }) => {
  return (
    <div className="bg-white rounded-2xl p-6 shadow-md border-t-4 border-indigo-500 relative">
      <div className="flex justify-between items-start mb-2">
        <span className="text-xs font-bold text-indigo-400 uppercase tracking-widest">
          Practice Phrase
        </span>
        <div className="flex gap-2">
          <button 
            onClick={onPrevious} 
            className="p-1 hover:bg-slate-100 rounded"
          >
            <ChevronLeft className="w-5 h-5 text-slate-400" />
          </button>
          <button 
            onClick={onNext} 
            className="p-1 hover:bg-slate-100 rounded"
          >
            <ChevronRight className="w-5 h-5 text-slate-400" />
          </button>
        </div>
      </div>
      <p className="text-2xl md:text-3xl font-medium text-slate-800 leading-snug">
        "{phrases.length > 0 ? phrases[currentIndex]?.text : "Loading..."}"
      </p>
    </div>
  );
};

export default PracticePhrase;
