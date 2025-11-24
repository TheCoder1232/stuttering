import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
import re
import logging

# Setup Module Logger
logger = logging.getLogger("SpeechCorrector")

class SpeechCorrector:
    def __init__(self, model_name="facebook/wav2vec2-large-960h"):
        """
        Initializes the model and processor.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CTC Model ({model_name}) on {self.device}...")
        
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            self.model.to(self.device)
            self.target_sample_rate = 16000
            logger.info("CTC Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading CTC Model: {e}", exc_info=True)
            self.model = None

    def correct_stutter_text(self, text):
        """
        Restores the regex-based correction logic to clean up the transcript.
        """
        if not text: return ""
        # Remove Word Repetitions (e.g., "I I I am")
        text = re.sub(r'\b(\w+)(?:[ -]\1)+\b', r'\1', text, flags=re.IGNORECASE)
        # Remove partial word cutoffs (e.g., "un- university")
        text = re.sub(r'\b\w-(?=\w)', '', text)
        # Clean extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def transcribe_chunk(self, audio_chunk):
        """
        Processes a single chunk of audio using CTC.
        """
        if self.model is None: return ""

        # Process input values
        inputs = self.processor(
            audio_chunk, 
            sampling_rate=self.target_sample_rate, 
            return_tensors="pt", 
            padding=True
        )

        # Move inputs to device
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        # CTC Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription.lower()

    def transcribe(self, audio_path, chunk_duration_s=10):
        """
        Loads audio, splits into chunks, transcribes, and corrects text.
        Returns: (raw_transcript, corrected_transcript)
        """
        if not os.path.exists(audio_path):
            logger.error(f"File not found: {audio_path}")
            return "", ""

        try:
            # Load audio using librosa (resamples automatically)
            full_audio, _ = librosa.load(audio_path, sr=self.target_sample_rate)
            
            full_transcription = []
            
            # Define chunk size in samples
            chunk_size = chunk_duration_s * self.target_sample_rate
            
            # Iterate through audio in chunks
            for i in range(0, len(full_audio), chunk_size):
                chunk = full_audio[i : i + chunk_size]
                
                # Skip extremely small chunks (<0.1s)
                if len(chunk) < 1600: 
                    continue
                    
                text = self.transcribe_chunk(chunk)
                full_transcription.append(text)

            # Join all parts for the raw transcript
            raw_text = " ".join(full_transcription)
            
            # Apply regex correction for the cleaned text
            corrected_text = self.correct_stutter_text(raw_text)

            return raw_text, corrected_text

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return "Error processing audio", "Error"

if __name__ == "__main__":
    # Test script
    engine = SpeechCorrector()
    # Replace with a dummy path for testing if needed
    print("Engine initialized.")