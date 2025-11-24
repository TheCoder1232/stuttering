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
from tts_engine import FluencyGenerator # <--- NEW

# --- CONFIGURATION ---
MODEL_PATH = r"../../models/stutter_model_epoch_10.pth" 
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "static_outputs" # <--- New folder for serving audio
CONFIDENCE_THRESHOLD = 0.50 

# Logging setup (Same as before)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Backend")

# Suppress noisy asyncio errors on Windows (connection reset by client)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

app = FastAPI(title="FluencyFlow Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DIRECTORY SETUP ---
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get absolute path for OUTPUT_DIR for logging
OUTPUT_DIR_ABS = os.path.abspath(OUTPUT_DIR)
logger.info(f"Static files directory: {OUTPUT_DIR_ABS}")

# Initialize Engines
stutter_engine = StutterPredictor(MODEL_PATH)
# transcription_engine = SpeechCorrector("openai/whisper-tiny.en")
transcription_engine = SpeechCorrector("facebook/wav2vec2-large-960h")
tts_engine = FluencyGenerator() # <--- Init TTS

# --- DATA MODELS ---
class StutterEvent(BaseModel):
    type: str
    start: float
    end: float
    confidence: float

class AnalysisResponse(BaseModel):
    filename: str
    original_audio_url: str # <--- URL for frontend to play
    corrected_audio_url: str # <--- URL for frontend to play
    original_transcript: str
    corrected_transcript: str
    fluency_score: int
    events: List[StutterEvent]
    feedback: List[str]

# --- HELPERS ---
def convert_to_wav(input_path):
    # (Same as your previous code)
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

# --- ENDPOINTS ---
@app.get("/phrases")
def get_practice_phrases():
    try:
        with open("phrases.json", "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        # Fallback if file is missing
        return [{"category": "Default", "text": "Hello, how are you today?"}]

@app.get("/static/{filename}")
@app.head("/static/{filename}")
async def serve_audio(filename: str):
    """Serve audio files with proper CORS headers (handles both GET and HEAD)"""
    from fastapi import Request
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    logger.info(f"Static file request ({request.method if 'request' in locals() else 'GET/HEAD'}): {filename}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type
    if filename.endswith('.wav'):
        media_type = "audio/wav"
    elif filename.endswith('.mp3'):
        media_type = "audio/mpeg"
    else:
        media_type = "application/octet-stream"
    
    logger.info(f"Serving {filename} as {media_type}")
    
    return FileResponse(
        file_path, 
        media_type=media_type,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.options("/static/{filename}")
async def serve_audio_options(filename: str):
    """Handle CORS preflight for audio files"""
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
    
    # Paths for serving files
    original_filename = f"{file_id}.wav"
    corrected_filename = f"{file_id}_corrected.mp3"
    
    public_original_path = os.path.join(OUTPUT_DIR, original_filename)
    public_corrected_path = os.path.join(OUTPUT_DIR, corrected_filename)

    logger.info(f"Processing: {file.filename} (ID: {file_id})")

    try:
        # 1. Save Raw File
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Convert to WAV
        # running in executor because disk I/O or conversion can be blocking
        loop = asyncio.get_running_loop()
        clean_temp_path = await loop.run_in_executor(None, convert_to_wav, temp_path)
        
        if not clean_temp_path: raise HTTPException(500, "Audio conversion failed")

        # 3. Move clean WAV to static folder
        shutil.move(clean_temp_path, public_original_path)

        # --- FIX: RUN HEAVY ML TASKS IN THREADPOOL ---
        
        # 4. Analyze Stutter (Blocking Call -> Thread)
        logger.info("Running Stutter Detection...")
        raw_events = await loop.run_in_executor(None, stutter_engine.predict, public_original_path)
        events = [e for e in raw_events if e['confidence'] >= CONFIDENCE_THRESHOLD]

        # 5. Transcribe (Blocking Call -> Thread)
        logger.info("Running Transcription...")
        transcript, corrected_text = await loop.run_in_executor(
            None, 
            transcription_engine.transcribe, 
            public_original_path
        )
        # ---------------------------------------------

        # 6. Generate Corrected Audio (TTS)
        # Assuming tts_engine.generate is already async? 
        # If it is NOT async, wrap it in run_in_executor too.
        # Based on your previous code, it looked like `await tts_engine.generate`
        logger.info("Generating Corrected Audio...")
        await tts_engine.generate(corrected_text, public_corrected_path)

        # 7. Scoring
        penalty = sum([min((e['end'] - e['start']) * 10, 10) for e in events])
        score = max(0, int(100 - penalty))

        # 8. Return URLs
        base_url = "http://localhost:8000/static"
        
        return {
            "filename": file.filename,
            "original_audio_url": f"{base_url}/{original_filename}",
            "corrected_audio_url": f"{base_url}/{corrected_filename}",
            "original_transcript": transcript,
            "corrected_transcript": corrected_text,
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
    # Configure uvicorn to suppress connection reset errors
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )