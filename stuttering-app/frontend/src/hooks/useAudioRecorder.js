import { useState, useRef } from "react";

export const useAudioRecorder = (onRecordingComplete) => {
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async (e) => {
    e.stopPropagation();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      
      recorder.ondataavailable = (e) => audioChunksRef.current.push(e.data);
      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        onRecordingComplete(audioBlob);
      };
      
      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecording(true);
    } catch (err) {
      throw new Error("Microphone access denied.");
    }
  };

  const stopRecording = (e) => {
    e.stopPropagation();
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const resetRecorder = () => {
    audioChunksRef.current = [];
    setRecording(false);
  };

  return {
    recording,
    startRecording,
    stopRecording,
    resetRecorder
  };
};
