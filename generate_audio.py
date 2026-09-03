"""
Sign Language Recognition - Offline Audio Generator (RUN ONCE, needs internet)
====================================================================================
Idhu script, ovvoru sign word ku, ovvoru language la (Tamil, Malayalam, Telugu,
Kannada, Hindi, English) audio file (.mp3) generate pannum, "audio" folder la save pannum.

IMPORTANT: Idha ORU THADAVAI mattum run pannunga, internet connection venum.
Idhukku apparam, main app internet illama, idhe pre-generated files ah play pannum.

Run pannuradhu:
    python generate_audio.py
"""

from gtts import gTTS
import os
import time
from translations import SIGN_TRANSLATIONS, LANGUAGES

OUTPUT_DIR = "audio"

os.makedirs(OUTPUT_DIR, exist_ok=True)

total = len(SIGN_TRANSLATIONS) * len(LANGUAGES)
done = 0
failed = []

print(f"Total audio files to generate: {total}")
print("Idhu konjam nerama edukkum (internet speed depend aagum)...\n")

for word, translations in SIGN_TRANSLATIONS.items():
    safe_name = word.replace(" ", "_")

    for lang_code in LANGUAGES.keys():
        lang_folder = os.path.join(OUTPUT_DIR, lang_code)
        os.makedirs(lang_folder, exist_ok=True)

        file_path = os.path.join(lang_folder, f"{safe_name}.mp3")

        # Already irundha skip pannurom (script mீண்டும் run pண்ணினா, waste aagama)
        if os.path.exists(file_path):
            done += 1
            continue

        text = translations.get(lang_code, word)

        try:
            tts = gTTS(text=text, lang=lang_code)
            tts.save(file_path)
            done += 1
            print(f"[{done}/{total}] Saved: {lang_code}/{safe_name}.mp3  ('{text}')")
            time.sleep(0.3)  # API ku romba fast ah requests pogama irukka
        except Exception as e:
            failed.append((word, lang_code, str(e)))
            print(f"FAILED: {word} ({lang_code}) - {e}")

print(f"\n{'='*50}")
print(f"Done! {done}/{total} audio files generated in '{OUTPUT_DIR}' folder.")
if failed:
    print(f"\n{len(failed)} files FAILED:")
    for word, lang, err in failed:
        print(f"  - {word} ({lang}): {err}")
    print("\nIdha mீண்டும் script run panni retry pannalam (already success aana files skip aagum).")
else:
    print("Ellame successful! Ippo app offline ah use pannalam.")
