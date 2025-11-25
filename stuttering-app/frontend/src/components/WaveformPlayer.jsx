import React, { useState, useRef, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";
import { Play, Pause } from "lucide-react";

const WaveformPlayer = ({ audioUrl, regions = [], label, color = "#4f46e5", onReady, seekTimestamp }) => {
  const containerRef = useRef(null);
  const waveSurferRef = useRef(null);
  const regionsPluginRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    if (!audioUrl) return;

    const initWaveSurfer = async () => {
      if (waveSurferRef.current) waveSurferRef.current.destroy();

      setIsReady(false);
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
        plugins: [wsRegions],
        interact: true,
        normalize: true,
        url: audioUrl,
      });

      ws.on("decode", () => {
        setIsReady(true);
        if (onReady) onReady(ws.getDuration());
      });

      ws.on("play", () => setIsPlaying(true));
      ws.on("pause", () => setIsPlaying(false));
      ws.on("finish", () => setIsPlaying(false));
      ws.on("interaction", () => ws.play());

      waveSurferRef.current = ws;
    };

    initWaveSurfer();

    return () => waveSurferRef.current?.destroy();
  }, [audioUrl, color, onReady]);

  useEffect(() => {
    if (regionsPluginRef.current && isReady && regions.length > 0) {
      regionsPluginRef.current.clearRegions();
      
      regions.forEach((event, index) => {
        regionsPluginRef.current.addRegion({
          start: event.start,
          end: event.end,
          color: 'rgba(239, 68, 68, 0.2)',
          drag: false,
          resize: false,
        });
      });
    }
  }, [regions, isReady]);

  useEffect(() => {
    if (seekTimestamp?.timestamp !== undefined && waveSurferRef.current && isReady) {
      waveSurferRef.current.setTime(seekTimestamp.timestamp);
      waveSurferRef.current.play();
    }
  }, [seekTimestamp, isReady]);

  return (
    <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm mb-4">
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs font-bold uppercase text-slate-400 tracking-wider">{label}</span>
        <button 
          onClick={() => waveSurferRef.current?.playPause()} 
          className="p-2 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-700 transition"
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </button>
      </div>
      <div ref={containerRef} className="w-full" />
    </div>
  );
};

export default WaveformPlayer;
