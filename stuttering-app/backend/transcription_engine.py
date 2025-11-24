import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import re
import librosa

class SpeechCorrector:
    def __init__(self, model_name="openai/whisper-tiny.en"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SpeechCorrector] Loading Whisper ({model_name}) on {self.device}...")
        
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Fix deprecated config warnings
            self.model.config.forced_decoder_ids = None
            self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
            print("[SpeechCorrector] Loaded successfully.")
        except Exception as e:
            print(f"[SpeechCorrector] Error loading Whisper: {e}")
            self.model = None

    def correct_stutter_text(self, text):
        """
        Rule-based regex correction for stuttered text.
        Example: "I I I want to..." -> "I want to..."
        """
        if not text: return ""
        # 1. Remove whole word repetitions: "the the the" -> "the"
        text = re.sub(r'\b(\w+)(?:[ -]\1)+\b', r'\1', text, flags=re.IGNORECASE)
        
        # 2. Remove partial word repetitions (e.g., "b-b-ball") if represented with hyphens
        text = re.sub(r'\b\w-(?=\w)', '', text)
        
        # 3. Clean extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def transcribe(self, audio_path):
        if self.model is None:
            return "Error: Model not loaded", "Error"

        try:
            # Load and resample to 16k
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Prepare input features
            input_features = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device)

            # Generate tokens
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            # Decode
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Apply correction
            corrected_text = self.correct_stutter_text(transcription)
            
            return transcription, corrected_text

        except Exception as e:
            print(f"Transcription error: {e}")
            return "", ""