import React from "react";
import WaveformPlayer from "./WaveformPlayer";
import AnnotatedTranscript from "./AnnotatedTranscript";

const ResultsSection = ({ results, audioDuration, setAudioDuration, seekRequest, handleSeek }) => {
  if (!results) return null;

  return (
    <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
      <div className="flex items-center justify-between bg-indigo-900 text-white p-6 rounded-xl shadow-lg">
        <div>
          <h2 className="text-sm opacity-80 uppercase tracking-wider">Fluency Score</h2>
          <div className="text-4xl font-bold">{results.fluency_score}/100</div>
        </div>
        <div className="text-right">
          <div className="text-sm opacity-80 uppercase tracking-wider">Events</div>
          <div className="text-2xl font-bold">{results.events.length}</div>
        </div>
      </div>

      <section>
        <WaveformPlayer 
          audioUrl={results.original_audio_url} 
          regions={results.events} 
          label="Your Recording" 
          color="#6366f1"
          onReady={(dur) => setAudioDuration(dur)}
          seekTimestamp={seekRequest}
        />
        
        <AnnotatedTranscript 
          text={results.original_transcript} 
          events={results.events} 
          duration={audioDuration}
          onSeek={handleSeek}
        />

        <WaveformPlayer 
          audioUrl={results.corrected_audio_url} 
          label="Corrected Flow" 
          color="#10b981"
        />
        
        <div className="bg-green-50 p-6 rounded-xl border border-green-200 text-slate-800">
          <span className="text-xs font-bold text-green-600 uppercase block mb-2">Target Transcript</span>
          {results.corrected_transcript}
        </div>
      </section>
    </div>
  );
};

export default ResultsSection;
