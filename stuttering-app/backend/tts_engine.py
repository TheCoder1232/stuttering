import edge_tts
import logging
import os

logger = logging.getLogger("TTSEngine")

class FluencyGenerator:
    def __init__(self, voice="en-US-ChristopherNeural"):
        # "en-US-ChristopherNeural" is a calm, male voice. 
        # "en-US-AriaNeural" is a good female alternative.
        self.voice = voice

    async def generate(self, text, output_path):
        """
        Generates clean audio from corrected text.
        """
        try:
            logger.info(f"Generating TTS for: '{text[:30]}...'")
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)
            return True
        except Exception as e:
            logger.error(f"TTS Generation failed: {e}")
            return False