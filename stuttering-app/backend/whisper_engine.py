import torch
import whisper
import os
import logging

logger = logging.getLogger("WhisperTranscriber")

class WhisperTranscriber:
    def __init__(self, model_name="tiny.en"):
        """
        Initializes the Whisper model for clean transcription.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper Model ({model_name}) on {self.device}...")
        
        try:
            self.model = whisper.load_model(model_name, device=self.device)
            logger.info("Whisper Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Whisper Model: {e}", exc_info=True)
            self.model = None

    def transcribe(self, audio_path):
        """
        Transcribes audio file using Whisper.
        Returns clean transcript text.
        """
        if self.model is None:
            logger.error("Whisper model not loaded")
            return ""

        if not os.path.exists(audio_path):
            logger.error(f"File not found: {audio_path}")
            return ""

        try:
            result = self.model.transcribe(audio_path, fp16=(self.device == "cuda"))
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}", exc_info=True)
            return "Error processing audio"

if __name__ == "__main__":
    engine = WhisperTranscriber()
    print("Whisper engine initialized.")
