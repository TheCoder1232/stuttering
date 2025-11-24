import sys
import os
import torch
import pyttsx3
import re
import soundfile as sf
import librosa 
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def correct_stutter(text):
    # Basic rule-based cleanup
    text = re.sub(r'\b(\w+)(?:[ -]\1){1,}\b', r'\1', text) 
    text = re.sub(r'\{.*?\}', '', text) 
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main(audio_path):
    print(f"Processing: {audio_path}")

    # 1. Load Whisper tiny.en (English specific model)
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # FIX 1: Clear deprecated forced_decoder_ids to suppress warnings
    model.config.forced_decoder_ids = None
    
    # FIX 2: Explicitly set pad_token_id to suppress attention mask warning
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    # 2. Load audio
    try:
        audio, sr = sf.read(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # 3. Resample if necessary
    if sr != 16000:
        # print(f"Resampling audio from {sr}Hz to 16000Hz...") # Optional: Comment out to reduce noise
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # 4. Prepare input
    input_features = processor(
        audio, 
        sampling_rate=sr, 
        return_tensors="pt"
    ).input_features.to(device)

    # 5. Generate transcription
    print("Transcribing...")
    with torch.no_grad():
        # CRITICAL FIX: Removed language="en" and task="transcribe"
        # The .en model implies these automatically.
        predicted_ids = model.generate(input_features)
        
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"\nOriginal Transcription: {transcription}")

    # 6. Correct
    corrected = correct_stutter(transcription)
    print(f"Corrected Sentence:     {corrected}")

    # 7. TTS output
    try:
        engine = pyttsx3.init()
        engine.say(corrected)
        engine.runAndWait()
    except RuntimeError:
        print("Error: Could not initialize TTS engine.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detection_correction.py <path_to_wav_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Error: File '{audio_file}' not found.")
        sys.exit(1)
        
    main(audio_file)