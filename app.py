"""
Sign Language Recognition - Full App
========================================
Idhu final app: real-time sign detection + sentence building + text-to-speech.

CONTROLS:
- Sign ah 1.5 seconds hold pannunga -> automatic ah sentence la add aagum
- 'c' key -> sentence clear pannum
- 'v' key -> sentence ah voice (audio) ah sollum
- 'b' key -> last word ah sentence la irundhu remove pannum (backspace)
- 'q' key -> exit pannum
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pickle
import numpy as np
import os
import time
from gtts import gTTS
import threading

MODEL_PATH = "model/sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"
CONFIDENCE_THRESHOLD = 70.0     # idha kammi konda, wrong predictions accept aagum
HOLD_DURATION = 1.5             # evlo seconds hold pannanum, sentence la add aaga

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: '{MODEL_PATH}' kedaikala. Munnadi 'train_model.py' run pannunga.")
    exit(1)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    running_mode=vision.RunningMode.IMAGE
)
landmarker = vision.HandLandmarker.create_from_options(options)


def normalize_landmarks(landmarks):
    base_x, base_y = landmarks[0][0], landmarks[0][1]
    normalized = []
    for x, y, z in landmarks:
        normalized.extend([x - base_x, y - base_y, z])
    return normalized


def draw_hand_landmarks(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start_idx, end_idx in connections:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)


def speak_text(text):
    """Background thread la audio generate panni play pannurom, app freeze aagama irukka."""
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("temp_speech.mp3")
        os.system("start temp_speech.mp3")  # Windows la default player la open aagum
    except Exception as e:
        print(f"Speech generation error: {e}")


def main():
    cap = cv2.VideoCapture(0)

    sentence = []
    current_word = ""
    hold_start_time = None
    last_added_word = ""

    print("App ready! Sign pannunga, 1.5 sec hold pannina sentence la add aagum.")
    print("Controls: c=clear, v=voice, b=backspace, q=quit")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Camera access panna mudiyala.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        prediction = None
        confidence = 0.0

        if result.hand_landmarks:
            draw_hand_landmarks(frame, result.hand_landmarks)
            landmark_list = [(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]]
            normalized = normalize_landmarks(landmark_list)
            input_data = np.array(normalized).reshape(1, -1)

            pred = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            conf = max(probabilities) * 100

            if conf >= CONFIDENCE_THRESHOLD:
                prediction = pred
                confidence = conf

        # Hold-to-confirm logic - sign ah steady ah hold pannina mattum sentence la add aagum
        if prediction:
            if prediction == current_word:
                elapsed = time.time() - hold_start_time
                if elapsed >= HOLD_DURATION and prediction != last_added_word:
                    sentence.append(prediction)
                    last_added_word = prediction
                    hold_start_time = time.time()  # reset, so same word again venumna hold pannanum
            else:
                current_word = prediction
                hold_start_time = time.time()
                last_added_word = ""  # puthu sign, so last_added reset pannurom
        else:
            current_word = ""
            hold_start_time = None
            last_added_word = ""

        # ===== UI drawing =====
        h, w, _ = frame.shape

        # Top bar - current prediction
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        if prediction:
            hold_progress = min((time.time() - hold_start_time) / HOLD_DURATION, 1.0) if hold_start_time else 0
            cv2.putText(frame, f"Detecting: {prediction} ({confidence:.0f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # Hold progress bar
            bar_width = int(300 * hold_progress)
            cv2.rectangle(frame, (10, 45), (310, 60), (100, 100, 100), 1)
            cv2.rectangle(frame, (10, 45), (10 + bar_width, 60), (0, 255, 255), -1)
        else:
            cv2.putText(frame, "Show a sign...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        # Bottom bar - sentence being built
        cv2.rectangle(frame, (0, h - 60), (w, h), (30, 30, 30), -1)
        sentence_text = " ".join(sentence) if sentence else "(sentence empty)"
        cv2.putText(frame, sentence_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Sign Language Recognition App', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = []
            print("Sentence cleared.")
        elif key == ord('b'):
            if sentence:
                removed = sentence.pop()
                print(f"Removed: {removed}")
        elif key == ord('v'):
            if sentence:
                full_text = " ".join(sentence)
                print(f"Speaking: {full_text}")
                threading.Thread(target=speak_text, args=(full_text,)).start()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
