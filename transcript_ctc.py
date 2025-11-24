import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os

class CTCTranscriber:
    def __init__(self, model_name="facebook/wav2vec2-large-960h"):
        """
        Initializes the model and processor.
        We use the 'large' model for better accuracy with specific speech patterns like stuttering.
        """
        print(f"Loading model: {model_name}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        self.target_sample_rate = 16000
        print(f"Model loaded successfully on {self.device.upper()}.")

    def load_audio(self, file_path):
        """
        Loads audio file and resamples it to 16kHz (required by Wav2Vec2).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # librosa loads audio as a float array and handles resampling
        speech_array, sampling_rate = librosa.load(file_path, sr=self.target_sample_rate)
        return speech_array

    def transcribe_chunk(self, audio_chunk):
        """
        Processes a single chunk of audio using CTC.
        """
        # Process input values
        inputs = self.processor(
            audio_chunk, 
            sampling_rate=self.target_sample_rate, 
            return_tensors="pt", 
            padding=True
        )

        # Move inputs to the same device as the model (GPU or CPU)
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            # Get logits (non-normalized predictions)
            logits = self.model(input_values).logits

        # CTC Decode: Take the argmax (highest probability character)
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the IDs back to text
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.lower()

    def process_file(self, file_path, chunk_duration_s=30):
        """
        Handles the logic for long files by splitting them into chunks.
        """
        print(f"Processing file: {file_path}")
        full_audio = self.load_audio(file_path)
        
        # Calculate total duration
        total_duration = len(full_audio) / self.target_sample_rate
        print(f"Total duration: {total_duration:.2f} seconds")

        full_transcription = []
        
        # Define chunk size in samples
        chunk_size = chunk_duration_s * self.target_sample_rate
        
        # Iterate through audio in chunks
        for i in range(0, len(full_audio), chunk_size):
            chunk = full_audio[i : i + chunk_size]
            
            # Pad the last chunk if it's too short (optional, but good for consistency)
            if len(chunk) < 1600: # Skip extremely small chunks (<0.1s)
                continue
                
            current_time = i / self.target_sample_rate
            print(f"Transcribing chunk starting at {current_time:.2f}s...")
            
            text = self.transcribe_chunk(chunk)
            full_transcription.append(text)

        # Join all parts
        final_text = " ".join(full_transcription)
        return final_text

if __name__ == "__main__":
    # Initialize the transcriber
    transcriber = CTCTranscriber()
    
    # --- CONFIGURATION ---
    # Replace this with your actual wav file path
    # If you don't have a file yet, the script will show an error but is ready to run.
    input_filename = r"E:\College\Final Y\Sem I\EDAI\stuttering\dataset\wav_files_16k\audio_0005_syllab_rep_clonic.wav" 
    
    try:
        # Check if file exists to prevent crash in demo
        if os.path.exists(input_filename):
            # UPDATED: Reduced chunk size to 10 seconds to fit 4GB VRAM GPUs (RTX 2050 Ti)
            result = transcriber.process_file(input_filename, chunk_duration_s=10)
            
            print("\n--- Final Transcription ---")
            print(result)
            
            # Save to text file
            with open("transcription_output.txt", "w") as f:
                f.write(result)
            print("\nSaved to transcription_output.txt")
        else:
            print(f"\n[!] Please place a .wav file named '{input_filename}' in this directory or update the script path.")
            
    except Exception as e:
        print(f"An error occurred: {e}")