"""
Neural TTS Module — Meta MMS-TTS (Massively Multilingual Speech)
--------------------------------------------------------------------
Much better quality than eSpeak NG for Indian languages. Runs fully
offline AFTER the one-time model download (needs internet only once
per language, first time it's used).

Install first:
    pip install transformers torch scipy

Models auto-download from Hugging Face the first time each language
is used (~100-150MB per language), then are cached locally forever
(no internet needed again for that language).
"""

import os
import tempfile
import numpy as np
import scipy.io.wavfile
from playsound import playsound

# Map your app's language codes -> Meta MMS-TTS model names
MMS_MODEL_MAP = {
    "ta": "facebook/mms-tts-tam",  # Tamil
    "hi": "facebook/mms-tts-hin",  # Hindi
    "te": "facebook/mms-tts-tel",  # Telugu
    "kn": "facebook/mms-tts-kan",  # Kannada
    "ml": "facebook/mms-tts-mal",  # Malayalam
}

# Cache loaded models in memory so we don't reload from disk every time
_model_cache = {}
_tokenizer_cache = {}


def _load_model(lang_code):
    from transformers import VitsModel, AutoTokenizer

    if lang_code not in _model_cache:
        model_name = MMS_MODEL_MAP[lang_code]
        print(f"[neural_tts] Loading model {model_name} (first time only)...")
        _model_cache[lang_code] = VitsModel.from_pretrained(model_name)
        _tokenizer_cache[lang_code] = AutoTokenizer.from_pretrained(model_name)
        print(f"[neural_tts] Model loaded and cached for '{lang_code}'.")

    return _model_cache[lang_code], _tokenizer_cache[lang_code]


def speak_neural(text, lang_code):
    """
    Speaks `text` using a high-quality neural voice for the given
    language. Falls back silently (prints error) if the language isn't
    supported or something goes wrong.
    """
    if lang_code not in MMS_MODEL_MAP:
        print(f"[neural_tts] No neural model configured for language '{lang_code}'.")
        return

    try:
        import torch

        model, tokenizer = _load_model(lang_code)

        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform

        waveform = output.squeeze().numpy()
        # Normalize to 16-bit PCM range for a standard .wav file
        waveform = np.clip(waveform, -1.0, 1.0)
        waveform_int16 = (waveform * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        scipy.io.wavfile.write(temp_path, rate=model.config.sampling_rate, data=waveform_int16)

        playsound(temp_path)
        try:
            os.remove(temp_path)
        except PermissionError:
            pass  # Windows sometimes briefly locks the file after playback — harmless, OS cleans temp files anyway

    except Exception as e:
        print(f"[neural_tts] Error speaking '{text}' in '{lang_code}': {e}")


if __name__ == "__main__":
    # Quick manual test
    speak_neural("என் பெயர் பிரதுஷா", "ta")
