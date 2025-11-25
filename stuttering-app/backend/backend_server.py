import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os
import uuid
import sys
import logging 
import json
import asyncio

# Engines
from inference_engine import StutterPredictor
from transcription_engine import SpeechCorrector
from whisper_engine import WhisperTranscriber
from tts_engine import FluencyGenerator

MODEL_PATH = r"../../models/stutter_model_epoch_10.pth" 
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "static_outputs"
CONFIDENCE_THRESHOLD = 0.50 

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logger = logging.getLogger("Backend")

app = FastAPI(title="FluencyFlow Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

stutter_engine = StutterPredictor(MODEL_PATH)
transcription_engine = SpeechCorrector("facebook/wav2vec2-large-960h")
# whisper_engine = WhisperTranscriber("base.en")
whisper_engine = WhisperTranscriber("small.en")
tts_engine = FluencyGenerator()

# --- DATA MODELS ---
class StutterEvent(BaseModel):
    type: str
    start: float
    end: float
    confidence: float

class AnalysisResponse(BaseModel):
    filename: str
    original_audio_url: str
    corrected_audio_url: str
    original_transcript: str
    corrected_transcript: str
    fluency_score: int
    events: List[StutterEvent]
    feedback: List[str]

def convert_to_wav(input_path):
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        output_path = input_path + "_clean.wav"
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        logger.error(f"FFmpeg Error: {e}")
        return None

def generate_feedback(events):
    # (Same as your previous code)
    feedback = []
    counts = {"block": 0, "prolongation": 0, "repetition": 0}
    for e in events:
        if e['type'] == 'block': counts['block'] += 1
        elif e['type'] == 'prolongation': counts['prolongation'] += 1
        elif 'rep' in e['type']: counts['repetition'] += 1

    if counts['prolongation'] > 0: feedback.append("Prolongation detected. Try 'Easy Onset'.")
    if counts['block'] > 0: feedback.append("Block detected. Try 'Pull-out' technique.")
    if counts['repetition'] > 0: feedback.append("Repetition detected. Use pacing.")
    if not feedback: feedback.append("Excellent fluency!")
    return feedback

@app.get("/phrases")
def get_practice_phrases():
    try:
        with open("phrases.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return [{"category": "Default", "text": "Hello, how are you today?"}]

@app.get("/static/{filename}")
@app.head("/static/{filename}")
async def serve_audio(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    media_type = "audio/wav" if filename.endswith('.wav') else "audio/mpeg" if filename.endswith('.mp3') else "application/octet-stream"
    
    return FileResponse(
        file_path, 
        media_type=media_type,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Cache-Control": "no-cache"
        }
    )

@app.options("/static/{filename}")
async def serve_audio_options(filename: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}_raw") 
    
    original_filename = f"{file_id}.wav"
    corrected_filename = f"{file_id}_corrected.mp3"
    
    public_original_path = os.path.join(OUTPUT_DIR, original_filename)
    public_corrected_path = os.path.join(OUTPUT_DIR, corrected_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        loop = asyncio.get_running_loop()
        clean_temp_path = await loop.run_in_executor(None, convert_to_wav, temp_path)
        
        if not clean_temp_path: raise HTTPException(500, "Audio conversion failed")

        shutil.move(clean_temp_path, public_original_path)

        raw_events = await loop.run_in_executor(None, stutter_engine.predict, public_original_path)
        events = [e for e in raw_events if e['confidence'] >= CONFIDENCE_THRESHOLD]

        transcript, corrected_text = await loop.run_in_executor(
            None, 
            transcription_engine.transcribe, 
            public_original_path
        )

        # Generate target transcript using Whisper
        target_transcript = await loop.run_in_executor(
            None,
            whisper_engine.transcribe,
            public_original_path
        )

        await tts_engine.generate(target_transcript, public_corrected_path)

        penalty = sum([min((e['end'] - e['start']) * 10, 10) for e in events])
        score = max(0, int(100 - penalty))

        base_url = "http://localhost:8000/static"
        
        return {
            "filename": file.filename,
            "original_audio_url": f"{base_url}/{original_filename}",
            "corrected_audio_url": f"{base_url}/{corrected_filename}",
            "original_transcript": transcript,
            "corrected_transcript": target_transcript,
            "fluency_score": score,
            "events": events,
            "feedback": generate_feedback(events)
        }

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")