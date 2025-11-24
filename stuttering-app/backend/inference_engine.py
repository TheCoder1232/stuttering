import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import torchaudio
from transformers import Wav2Vec2Model
import os
import logging

# Setup Module Logger
logger = logging.getLogger("InferenceEngine")

# --- 1. MODEL ARCHITECTURE ---
class StutterDetector(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", num_classes=5):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.wav2vec2.feature_extractor._freeze_parameters()
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_classes)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        logits = self.classifier(outputs.last_hidden_state)
        return logits

# --- 2. INFERENCE ENGINE ---
class StutterPredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model from {model_path} on {self.device}...")
        
        self.config = {
            'model_name': "facebook/wav2vec2-base",
            'num_classes': 5,
            'target_sr': 16000
        }
        
        self.id2label = {
            0: 'fluent', 1: 'block', 2: 'word_rep', 3: 'syllab_rep', 4: 'prolongation'
        }

        try:
            self.model = StutterDetector(self.config['model_name'], self.config['num_classes'])
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info("Model weights loaded successfully.")
            else:
                logger.warning(f"Checkpoint not found at {model_path}. Using random weights.")

            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.critical(f"Error loading model: {e}", exc_info=True)
            self.model = None

    def _preprocess_audio(self, file_path):
        try:
            audio_signal, sr = sf.read(file_path)
            waveform = torch.from_numpy(audio_signal).float()
            
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()
            
            target_sr = self.config['target_sr']
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
                
            if waveform.shape[0] > 1:
                waveform = waveform[0].unsqueeze(0)

            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
                
            return waveform.squeeze()

        except Exception as e:
            logger.error(f"Error loading/preprocessing audio: {e}")
            return None

    def predict(self, file_path):
        if self.model is None: 
            logger.error("Predict called but model is None")
            return []

        waveform = self._preprocess_audio(file_path)
        if waveform is None: return []

        target_sr = self.config['target_sr']
        window_size = target_sr * 5 
        stride = target_sr * 5      
        
        all_events = []
        total_samples = len(waveform)
        
        with torch.no_grad():
            for start_idx in range(0, total_samples, stride):
                end_idx = min(start_idx + window_size, total_samples)
                chunk = waveform[start_idx:end_idx]
                
                if len(chunk) < 1600: continue 
                    
                input_values = chunk.unsqueeze(0).to(self.device)
                logits = self.model(input_values)

                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidences, preds = torch.max(probs, dim=-1)
                
                preds = preds.squeeze().cpu().numpy()
                confidences = confidences.squeeze().cpu().numpy()
                
                model_stride_samples = 320 
                current_event = None
                
                for i, label_id in enumerate(preds):
                    label_name = self.id2label.get(label_id, 'unknown')
                    score = float(confidences[i])
                    timestamp = (start_idx + (i * model_stride_samples)) / float(target_sr)
                    
                    if label_id == 0:
                        if current_event:
                            current_event['end'] = timestamp
                            all_events.append(current_event)
                            current_event = None
                        continue
                    
                    if current_event is None:
                        current_event = {
                            'type': label_name, 
                            'start': timestamp, 
                            'end': timestamp + 0.02,
                            'scores': [score]
                        }
                    elif current_event['type'] != label_name:
                        current_event['end'] = timestamp
                        all_events.append(current_event)
                        current_event = {
                            'type': label_name, 
                            'start': timestamp, 
                            'end': timestamp + 0.02,
                            'scores': [score]
                        }
                    else:
                        current_event['scores'].append(score)
                
                if current_event:
                    current_event['end'] = (start_idx + (len(preds) * model_stride_samples)) / float(target_sr)
                    all_events.append(current_event)

        return self._merge_events(all_events)

    def _merge_events(self, events, tolerance=0.2):
        if not events: return []
        merged = [events[0]]
        
        for curr in events[1:]:
            prev = merged[-1]
            if curr['type'] == prev['type'] and (curr['start'] - prev['end'] < tolerance):
                prev['end'] = curr['end']
                prev['scores'].extend(curr['scores'])
            else:
                merged.append(curr)
        
        final_events = []
        for ev in merged:
            avg_score = sum(ev['scores']) / len(ev['scores'])
            final_events.append({
                'type': ev['type'],
                'start': ev['start'],
                'end': ev['end'],
                'confidence': round(avg_score, 4)
            })
            
        return final_events