import React, { useState, useEffect } from "react";
import { Volume2, RefreshCw } from "lucide-react";
import ErrorBoundary from "./components/ErrorBoundary";
import PracticePhrase from "./components/PracticePhrase";
import RecordingZone from "./components/RecordingZone";
import ResultsSection from "./components/ResultsSection";
import { useAudioRecorder } from "./hooks/useAudioRecorder";
const SpeechTherapistApp = () => {
  const [phrases, setPhrases] = useState([]);
  const [currentPhraseIdx, setCurrentPhraseIdx] = useState(0);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [audioDuration, setAudioDuration] = useState(0);
  const [seekRequest, setSeekRequest] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/phrases")
      .then(res => res.json())
      .then(data => setPhrases(data))
      .catch(err => console.error("Failed to load phrases", err));
  }, []);

  const processAudioFile = async (audioBlob) => {
    setAnalyzing(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", audioBlob, audioBlob.name || "recording.webm");

    try {
      const res = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Analysis failed");
      
      setResults(data);
    } catch (err) {
      setError(err.message || "Unknown error occurred");
    } finally {
      setAnalyzing(false);
    }
  };

  const { recording, startRecording, stopRecording, resetRecorder } = useAudioRecorder(processAudioFile);

  const resetSession = () => {
    setResults(null);
    setError(null);
    setAnalyzing(false);
    setAudioDuration(0);
    setSeekRequest(null);
    resetRecorder();
  };

  const onDrop = (acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      resetSession();
      processAudioFile(acceptedFiles[0]);
    }
  };

  const handleSeek = (time) => setSeekRequest({ timestamp: time, key: Date.now() });

  return (
    <div className="min-h-screen bg-slate-50 p-6 md:p-8 font-sans text-slate-900">
      <div className="max-w-5xl mx-auto">
        <header className="mb-8 flex justify-between items-center">
          <h1 className="text-3xl font-bold text-indigo-700 flex items-center gap-2">
            <Volume2 className="h-8 w-8" /> FluencyFlow
          </h1>
          {results && (
            <button 
                onClick={resetSession}
                className="flex items-center gap-2 px-4 py-2 bg-white border border-indigo-200 text-indigo-600 rounded-full hover:bg-indigo-50 transition shadow-sm font-medium text-sm"
            >
                <RefreshCw className="w-4 h-4" /> New Session
            </button>
          )}
        </header>

        {error && (
          <div className="mb-6 p-4 bg-red-100 text-red-700 rounded-lg border border-red-200 flex items-center gap-2">
            <span className="font-bold">Error:</span> {error}
          </div>
        )}

        <div className="grid gap-6">
          <PracticePhrase
            phrases={phrases}
            currentIndex={currentPhraseIdx}
            onPrevious={() => setCurrentPhraseIdx(p => Math.max(0, p - 1))}
            onNext={() => setCurrentPhraseIdx(p => Math.min(phrases.length - 1, p + 1))}
          />

          <RecordingZone
            recording={recording}
            analyzing={analyzing}
            results={results}
            onStartRecording={startRecording}
            onStopRecording={stopRecording}
            onDrop={onDrop}
          />

          <ResultsSection
            results={results}
            audioDuration={audioDuration}
            setAudioDuration={setAudioDuration}
            seekRequest={seekRequest}
            handleSeek={handleSeek}
          />
        </div>
      </div>
    </div>
  );
};

const App = () => (
  <ErrorBoundary>
    <SpeechTherapistApp />
  </ErrorBoundary>
);

export default App;