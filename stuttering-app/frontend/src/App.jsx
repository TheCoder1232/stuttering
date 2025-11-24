import React, { useState, useRef, useEffect, useMemo } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";
import { useDropzone } from "react-dropzone";
import {
  Mic, Square, Loader2, Play, Pause, Volume2, 
  UploadCloud, RefreshCw, ChevronRight, ChevronLeft,
  Activity, Clock
} from "lucide-react";

// --- ERROR BOUNDARY ---
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("React Error Boundary caught:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-50 p-6 flex items-center justify-center">
          <div className="bg-red-50 border border-red-200 rounded-xl p-6 max-w-lg">
            <h2 className="text-xl font-bold text-red-700 mb-2">Something went wrong</h2>
            <p className="text-red-600 mb-4">{this.state.error?.message || "Unknown error"}</p>
            <button 
              onClick={() => window.location.reload()} 
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// --- COMPONENT: Waveform Player ---
// --- COMPONENT: WaveformPlayer (Fixed) ---
const WaveformPlayer = ({ audioUrl, regions = [], label, color = "#4f46e5", onReady, seekTimestamp }) => {
  const containerRef = useRef(null);
  const waveSurferRef = useRef(null);
  const regionsPluginRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loadError, setLoadError] = useState(null);
  const [useFallback, setUseFallback] = useState(false);
  const [audioReady, setAudioReady] = useState(false); // Tracks if WS is actually ready
  const audioRef = useRef(null);

  // 1. Initialize WaveSurfer (Run ONLY when audioUrl changes)
  useEffect(() => {
    if (!containerRef.current || !audioUrl || useFallback) return;

    // Cleanup previous instance if it exists (Double-render protection)
    if (waveSurferRef.current) {
      waveSurferRef.current.destroy();
    }

    const initWaveSurfer = async () => {
      try {
        const wsRegions = RegionsPlugin.create();
        regionsPluginRef.current = wsRegions;

        const ws = WaveSurfer.create({
          container: containerRef.current,
          waveColor: "#e2e8f0",
          progressColor: color,
          cursorColor: color,
          barWidth: 2,
          barGap: 3,
          height: 80,
          plugins: [wsRegions], // Register plugin here
          interact: true,
          normalize: true,
          url: audioUrl, // Load URL directly in config for v7 compatibility
        });

        // Event Listeners
        ws.on("decode", () => {
          setAudioReady(true);
          if (onReady) onReady(ws.getDuration());
        });

        ws.on("error", (err) => {
          console.error("WaveSurfer Error:", err);
          setLoadError(err.message);
          // Optional: Enable fallback automatically on specific errors
          // setUseFallback(true); 
        });

        ws.on("play", () => setIsPlaying(true));
        ws.on("pause", () => setIsPlaying(false));
        ws.on("finish", () => setIsPlaying(false));
        ws.on("interaction", () => ws.play());

        waveSurferRef.current = ws;
      } catch (err) {
        console.error("Init Error:", err);
        setUseFallback(true);
      }
    };

    initWaveSurfer();

    return () => {
      if (waveSurferRef.current) {
        waveSurferRef.current.destroy();
        waveSurferRef.current = null;
      }
    };
  }, [audioUrl, useFallback, color]); // Removed 'regions' from here!

  // 2. Handle Regions Updates (Run when regions change, WITHOUT destroying player)
  useEffect(() => {
    const ws = waveSurferRef.current;
    const wsRegions = regionsPluginRef.current;

    if (ws && wsRegions && audioReady) {
      console.log("Regions available but not adding to waveform:", regions.length);
      wsRegions.clearRegions();
      // Regions are not being added to the waveform visualization
    }
  }, [regions, audioReady]);

  // 3. Handle External Seek
  useEffect(() => {
    if (seekTimestamp?.timestamp !== undefined) {
      const time = seekTimestamp.timestamp;
      
      if (useFallback && audioRef.current) {
        audioRef.current.currentTime = time;
        audioRef.current.play().catch(console.error);
      } else if (waveSurferRef.current && audioReady) {
        waveSurferRef.current.setTime(time);
        waveSurferRef.current.play();
      }
    }
  }, [seekTimestamp, useFallback, audioReady]);

  const togglePlay = () => {
    if (useFallback && audioRef.current) {
      if (audioRef.current.paused) audioRef.current.play();
      else audioRef.current.pause();
    } else {
      waveSurferRef.current?.playPause();
    }
  };

  // --- RENDER FALLBACK ---
  if (useFallback) {
    return (
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm mb-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-xs font-bold uppercase text-slate-400 tracking-wider">{label} (Simple)</span>
          <button onClick={togglePlay} className="p-2 bg-slate-100 rounded-full">
             {isPlaying ? <Pause className="w-4 h-4"/> : <Play className="w-4 h-4"/>}
          </button>
        </div>
        <audio 
          ref={audioRef}
          src={audioUrl} 
          controls 
          className="w-full"
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onLoadedMetadata={(e) => onReady && onReady(e.target.duration)}
        />
      </div>
    );
  }

  // --- RENDER WAVESURFER ---
  return (
    <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm mb-4">
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs font-bold uppercase text-slate-400 tracking-wider">{label}</span>
        <button 
          onClick={togglePlay} 
          disabled={!audioReady}
          className={`p-2 rounded-full transition ${audioReady ? 'bg-slate-100 hover:bg-slate-200 text-slate-700' : 'bg-slate-50 text-slate-300'}`}
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </button>
      </div>
      
      {/* Error Message Display */}
      {loadError && (
        <div className="text-xs text-red-500 mb-2">
          Error loading visualizer. <button onClick={() => setUseFallback(true)} className="underline">Switch to simple player</button>
        </div>
      )}

      {/* The WaveSurfer Container */}
      <div ref={containerRef} className="w-full relative" />
    </div>
  );
};
// --- COMPONENT: Annotated Transcript & Event List ---
const AnnotatedTranscript = ({ text, events, duration, onSeek }) => {
  
  const processedContent = useMemo(() => {
    // Safeguard: Ensure text is a string
    if (!text || typeof text !== 'string') {
      console.warn("AnnotatedTranscript: Invalid text", text);
      return text || "(No transcript available)";
    }
    
    if (!events || events.length === 0 || !duration) return text;

    // --- IMPROVED MAPPING ALGORITHM ---
    // Instead of guessing CPS, we use the EXACT percentage of the file.
    // If an event is at 50% of the audio, we highlight the text at 50% of the string.
    const mapTimeToIndex = (time) => Math.floor((time / duration) * text.length);

    const ranges = events.map(event => ({
      startIndex: Math.max(0, mapTimeToIndex(event.start)),
      endIndex: Math.min(text.length, Math.max(mapTimeToIndex(event.start) + 1, mapTimeToIndex(event.end))),
      ...event
    })).sort((a, b) => a.startIndex - b.startIndex);

    let result = [];
    let lastIndex = 0;

    ranges.forEach((range, i) => {
      // A. Fluent Text
      if (range.startIndex > lastIndex) {
        result.push(
          <span key={`text-${i}`}>
            {text.substring(lastIndex, range.startIndex)}
          </span>
        );
      }

      // B. Highlighted Stutter (Clickable, but NO badge inside)
      const stutteredSegment = text.substring(range.startIndex, range.endIndex);
      result.push(
        <span 
          key={`highlight-${i}`}
          onClick={() => onSeek(range.start)}
          className="bg-red-100 text-red-800 border-b-2 border-red-300 px-0.5 rounded-sm cursor-pointer hover:bg-red-200 transition-colors"
          title={`Click to listen: ${range.type}`}
        >
          {stutteredSegment}
        </span>
      );

      lastIndex = range.endIndex;
    });

    if (lastIndex < text.length) {
      result.push(<span key="text-end">{text.substring(lastIndex)}</span>);
    }

    return result;
  }, [text, events, duration]);

  // If duration isn't loaded yet, show plain text
  if (!duration) return <div className="p-4 text-slate-400 italic">Syncing transcript...</div>;

  return (
    <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm mb-6 flex flex-col md:flex-row gap-6">
      
      {/* LEFT: Text Paragraph */}
      <div className="flex-1">
         <div className="flex justify-between items-center mb-3">
            <span className="text-xs font-bold text-slate-400 uppercase">Transcript</span>
         </div>
         <p className="text-lg text-slate-700 leading-loose font-medium font-serif">
           {processedContent}
         </p>
      </div>

      {/* RIGHT/BOTTOM: Event List (The user requested "at the end") */}
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
                        className="w-full text-left bg-slate-50 hover:bg-red-50 hover:border-red-200 border border-slate-200 rounded-lg p-3 transition group group"
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

// --- MAIN APP ---
const SpeechTherapistApp = () => {
  const [phrases, setPhrases] = useState([]);
  const [currentPhraseIdx, setCurrentPhraseIdx] = useState(0);

  const [recording, setRecording] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  // New state to coordinate audio and text
  const [audioDuration, setAudioDuration] = useState(0); 
  const [seekRequest, setSeekRequest] = useState(null); // {timestamp, key} to seek to

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  useEffect(() => {
    fetch("http://localhost:8000/phrases")
      .then(res => res.json())
      .then(data => setPhrases(data))
      .catch(err => console.error("Failed to load phrases", err));
  }, []);

  const resetSession = () => {
    setResults(null);
    setError(null);
    setAnalyzing(false);
    setRecording(false);
    setAudioDuration(0);
    setSeekRequest(null);
    audioChunksRef.current = [];
  };

  const processAudioFile = async (audioBlob) => {
    setAnalyzing(true);
    setError(null);
    const formData = new FormData();
    const filename = audioBlob.name || "recording.webm"; 
    formData.append("file", audioBlob, filename);

    try {
      console.log("Sending audio to backend...");
      const res = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });
      
      console.log("Response status:", res.status);
      const data = await res.json();
      console.log("Response data:", data);
      
      if (!res.ok) throw new Error(data.detail || "Analysis failed");
      
      // Validate response structure
      if (!data.original_audio_url || !data.corrected_audio_url) {
        throw new Error("Invalid response: missing audio URLs");
      }
      if (!data.original_transcript) {
        console.warn("Warning: No transcript returned");
      }
      if (!Array.isArray(data.events)) {
        console.warn("Warning: events is not an array, defaulting to empty");
        data.events = [];
      }
      
      console.log("Setting results:", data);
      setResults(data);
    } catch (err) {
      console.error("Error processing audio:", err);
      setError(err.message || "Unknown error occurred");
    } finally {
      setAnalyzing(false);
    }
  };

  const startRecording = async (e) => {
    e.stopPropagation();
    resetSession();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => audioChunksRef.current.push(e.data);
      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        processAudioFile(audioBlob);
      };
      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecording(true);
    } catch (err) {
      setError("Microphone access denied.");
    }
  };

  const stopRecording = (e) => {
    e.stopPropagation();
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const onDrop = (acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      resetSession();
      processAudioFile(acceptedFiles[0]);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'audio/*': [] },
    disabled: recording || analyzing || results
  });

  // Handler to trigger seek in WaveformPlayer
  const handleSeek = (time) => {
    setSeekRequest({ timestamp: time, key: Date.now() });
  };

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
          <div className="bg-white rounded-2xl p-6 shadow-md border-t-4 border-indigo-500 relative">
             <div className="flex justify-between items-start mb-2">
                <span className="text-xs font-bold text-indigo-400 uppercase tracking-widest">
                    Practice Phrase
                </span>
                <div className="flex gap-2">
                    <button onClick={() => setCurrentPhraseIdx(p => Math.max(0, p - 1))} className="p-1 hover:bg-slate-100 rounded"><ChevronLeft className="w-5 h-5 text-slate-400"/></button>
                    <button onClick={() => setCurrentPhraseIdx(p => Math.min(phrases.length - 1, p + 1))} className="p-1 hover:bg-slate-100 rounded"><ChevronRight className="w-5 h-5 text-slate-400"/></button>
                </div>
             </div>
             <p className="text-2xl md:text-3xl font-medium text-slate-800 leading-snug">
                "{phrases.length > 0 ? phrases[currentPhraseIdx]?.text : "Loading..."}"
             </p>
          </div>

          {!results && (
            <div 
                {...getRootProps()} 
                className={`relative rounded-2xl border-2 border-dashed transition-all duration-300 min-h-[300px] flex flex-col items-center justify-center p-8 cursor-pointer ${isDragActive ? "border-indigo-500 bg-indigo-50" : "border-slate-300 bg-white hover:border-indigo-300"} ${analyzing ? "pointer-events-none opacity-80" : ""}`}
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
                            onClick={recording ? stopRecording : startRecording}
                            className={`z-20 w-24 h-24 rounded-full flex items-center justify-center transition-all shadow-xl mb-6 ${recording ? "bg-red-500 hover:bg-red-600 animate-pulse" : "bg-indigo-600 hover:bg-indigo-700 hover:scale-105"}`}
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
          )}

          {results && (
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
                        onReady={(dur) => setAudioDuration(dur)} // 1. Get Duration
                        seekTimestamp={seekRequest} // 2. Listen for seek commands
                    />
                    
                    {/* Annotated Transcript now includes the Event List */}
                    <AnnotatedTranscript 
                        text={results.original_transcript} 
                        events={results.events} 
                        duration={audioDuration} // 3. Pass duration down
                        onSeek={handleSeek} // 4. Pass seek handler
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
          )}
        </div>
      </div>
    </div>
  );
};

// Wrap with Error Boundary for production
const App = () => (
  <ErrorBoundary>
    <SpeechTherapistApp />
  </ErrorBoundary>
);

export default App;