import React, { useMemo } from "react";
import { Activity, Clock } from "lucide-react";

const AnnotatedTranscript = ({ text, events, duration, onSeek }) => {
  const processedContent = useMemo(() => {
    if (!text || typeof text !== 'string') return text || "(No transcript available)";
    if (!events?.length || !duration) return text;

    const mapTimeToIndex = (time) => Math.floor((time / duration) * text.length);

    const ranges = events.map(event => ({
      startIndex: Math.max(0, mapTimeToIndex(event.start)),
      endIndex: Math.min(text.length, Math.max(mapTimeToIndex(event.start) + 1, mapTimeToIndex(event.end))),
      ...event
    })).sort((a, b) => a.startIndex - b.startIndex);

    let result = [];
    let lastIndex = 0;

    ranges.forEach((range, i) => {
      if (range.startIndex > lastIndex) {
        result.push(<span key={`text-${i}`}>{text.substring(lastIndex, range.startIndex)}</span>);
      }

      result.push(
        <span 
          key={`highlight-${i}`}
          onClick={() => onSeek(range.start)}
          className="bg-red-100 text-red-800 border-b-2 border-red-300 px-0.5 rounded-sm cursor-pointer hover:bg-red-200 transition-colors"
          title={`Click to listen: ${range.type}`}
        >
          {text.substring(range.startIndex, range.endIndex)}
        </span>
      );

      lastIndex = range.endIndex;
    });

    if (lastIndex < text.length) {
      result.push(<span key="text-end">{text.substring(lastIndex)}</span>);
    }

    return result;
  }, [text, events, duration, onSeek]);

  if (!duration) return <div className="p-4 text-slate-400 italic">Syncing transcript...</div>;

  return (
    <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm mb-6 flex flex-col md:flex-row gap-6">
      <div className="flex-1">
         <div className="flex justify-between items-center mb-3">
            <span className="text-xs font-bold text-slate-400 uppercase">Transcript</span>
         </div>
         <p className="text-lg text-slate-700 leading-loose font-medium font-serif">
           {processedContent}
         </p>
      </div>

      <div className="md:w-64 border-l md:border-l-slate-100 md:pl-6 border-t md:border-t-0 pt-6 md:pt-0">
        <h3 className="text-xs font-bold text-slate-400 uppercase mb-3 flex items-center gap-2">
            <Activity className="w-4 h-4" /> Dysfluency Report
        </h3>
        
        {events.length === 0 ? (
            <p className="text-sm text-green-600 font-medium">No stuttering detected!</p>
        ) : (
            <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                {events.map((event, idx) => (
                    <button
                        key={idx}
                        onClick={() => onSeek(event.start)}
                        className="w-full text-left bg-slate-50 hover:bg-red-50 hover:border-red-200 border border-slate-200 rounded-lg p-3 transition group"
                    >
                        <div className="flex justify-between items-start">
                            <span className="text-sm font-bold text-slate-700 capitalize group-hover:text-red-700">
                                {event.type.replace('_', ' ')}
                            </span>
                            <span className="text-[10px] bg-slate-200 text-slate-600 px-1.5 py-0.5 rounded flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {event.start.toFixed(1)}s
                            </span>
                        </div>
                        <div className="mt-1 flex justify-between items-center">
                            <span className="text-xs text-slate-400">Confidence</span>
                            <div className="w-16 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                                <div 
                                    className="h-full bg-green-500" 
                                    style={{ width: `${event.confidence * 100}%` }}
                                />
                            </div>
                        </div>
                    </button>
                ))}
            </div>
        )}
      </div>
    </div>
  );
};

export default AnnotatedTranscript;
