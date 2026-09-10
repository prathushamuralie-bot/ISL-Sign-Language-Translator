"""
Sign Language Recognition - Full App (Enhanced)
====================================================
Two-Hand Sign Recognition
+ Sentence Building
+ Text-to-Speech
+ Emergency Mode
+ Ambiguous Sign Disambiguation (NOW ACTIVE via real facial detection)
+ Facial Expression Detection (eyebrow raise -> QUESTION, head shake -> NEGATIVE, head nod -> POSITIVE)
+ Voice-to-Sign (offline speech recognition using Vosk)
+ Predictive Word Suggestions

MODEL:
- sign_data_two_hands.csv
- 126 features
- 21 landmarks x 3 coordinates x 2 hands

SETUP NEEDED BEFORE RUNNING:
    pip install vosk sounddevice
    Download face_landmarker.task from:
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    Download a Vosk model (small English) from https://alphacephei.com/vosk/models
        -> unzip it, rename the folder to "vosk-model", place in project root

CONTROLS:
- Sign ah 1.5 seconds hold pannunga -> sentence la add aagum
- 'c' -> Clear sentence
- 'v' -> Voice (speak the sentence)
- 'b' -> Last word remove
- 'e' -> Emergency reset
- 'm' -> Voice-to-Sign mode (listens to mic, shows recognized text on screen)
- '1' / '2' / '3' -> Pick a predictive word suggestion (when shown)
- 'q' -> Quit
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import pickle
import numpy as np
import os
import time
import json
import queue
from gtts import gTTS
import threading
from collections import deque


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "model/sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_PATH = "face_landmarker.task"
VOSK_MODEL_PATH = "vosk-model"

CONFIDENCE_THRESHOLD = 70.0
HOLD_DURATION = 1.5

# Difference between top two predictions.
# If difference is <= 15%, we consider it ambiguous.
AMBIGUITY_THRESHOLD = 15.0

# Facial expression thresholds
EYEBROW_RAISE_THRESHOLD = 0.4          # blendshape score (0-1)
HEAD_MOVEMENT_HISTORY_LEN = 15         # how many frames of head position to track
HEAD_SHAKE_STD_THRESHOLD = 0.012       # x-axis std dev -> shaking head (NEGATIVE)
HEAD_NOD_STD_THRESHOLD = 0.012         # y-axis std dev -> nodding head (POSITIVE)


# ============================================================
# EMERGENCY SIGNS
# ============================================================

EMERGENCY_SIGNS = {
    "help": "I need help",
    "doctor": "I need a doctor",
    "emergency": "Emergency",
    "family": "Please call my family",
    "ambulance": "Please call an ambulance"
}


# ============================================================
# AMBIGUOUS SIGN GROUPS
# ============================================================
#
# IMPORTANT:
# Unga model labels exact-a different-a irundha
# inga labels-ai change pannunga.
#

AMBIGUOUS_SIGN_GROUPS = [
    {"can", "can't"},
    {"yes", "no"},
    {"good", "bad"},
    {"come", "go"},
    {"here", "there"},
    {"this", "that"}
]


# ============================================================
# PREDICTIVE WORD SUGGESTIONS
# ============================================================
#
# After a word is added to the sentence, we look up likely next
# words here and show them on screen. User can pick one instantly
# with number keys 1/2/3 instead of signing it.
#
# IMPORTANT: Adjust these to match your model's actual vocabulary.
#

NEXT_WORD_SUGGESTIONS = {
    "i": ["am", "need", "want"],
    "you": ["are", "help", "name"],
    "my": ["name", "family", "help"],
    "name": ["is", "what"],
    "is": ["good", "bad", "here"],
    "i need": ["help", "doctor", "water"],
    "please": ["help", "call", "come"],
    "can": ["you", "help", "i"],
    "help": ["me", "please"],
    "call": ["doctor", "ambulance", "family"],
}


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: '{MODEL_PATH}' kedaikala.")
    print("Munnadi train_model.py run pannunga.")
    exit(1)

if not os.path.exists(HAND_MODEL_PATH):
    print(f"ERROR: '{HAND_MODEL_PATH}' kedaikala.")
    print("hand_landmarker.task file project folder-la irukkanum.")
    exit(1)

FACE_DETECTION_ENABLED = os.path.exists(FACE_MODEL_PATH)
if not FACE_DETECTION_ENABLED:
    print(f"WARNING: '{FACE_MODEL_PATH}' kedaikala. Facial expression detection OFF.")
    print("Download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")

VOICE_TO_SIGN_ENABLED = os.path.exists(VOSK_MODEL_PATH)
if not VOICE_TO_SIGN_ENABLED:
    print(f"WARNING: '{VOSK_MODEL_PATH}' folder kedaikala. Voice-to-Sign mode OFF.")
    print("Download a model from https://alphacephei.com/vosk/models, unzip, rename folder to 'vosk-model'.")


# ============================================================
# LOAD MODEL
# ============================================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    running_mode=vision.RunningMode.IMAGE
)

landmarker = vision.HandLandmarker.create_from_options(options)


# ============================================================
# MEDIAPIPE FACE LANDMARKER (for facial expression detection)
# ============================================================

face_landmarker = None

if FACE_DETECTION_ENABLED:
    face_base_options = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        running_mode=vision.RunningMode.IMAGE
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options)


# ============================================================
# VOSK (offline voice-to-sign)
# ============================================================

vosk_model = None

if VOICE_TO_SIGN_ENABLED:
    from vosk import Model as VoskModel, KaldiRecognizer
    import sounddevice as sd

    vosk_model = VoskModel(VOSK_MODEL_PATH)


# ============================================================
# GET HAND LANDMARKS
# ============================================================

def get_hand_landmarks(result):
    """
    MediaPipe result-la irundhu
    LEFT + RIGHT hand landmarks edukkum.

    Each hand:
        21 landmarks x 3 = 63

    Two hands:
        63 + 63 = 126
    """
    left_hand = None
    right_hand = None

    if not result.hand_landmarks:
        return left_hand, right_hand

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        if result.handedness:
            handedness = result.handedness[i][0].category_name
            if handedness.lower() == "left":
                left_hand = hand_landmarks
            elif handedness.lower() == "right":
                right_hand = hand_landmarks
        else:
            # Fallback
            if left_hand is None:
                left_hand = hand_landmarks
            else:
                right_hand = hand_landmarks

    return left_hand, right_hand


# ============================================================
# CONVERT HAND TO FEATURES
# ============================================================

def hand_to_features(hand_landmarks):
    """
    One hand:
    21 landmarks x 3 = 63 values
    """
    if hand_landmarks is None:
        return [0.0] * 63

    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y

    features = []
    for lm in hand_landmarks:
        features.extend([lm.x - base_x, lm.y - base_y, lm.z])

    return features


# ============================================================
# CREATE 126 FEATURES
# ============================================================

def create_two_hand_features(left_hand, right_hand):
    """
    LEFT  = 63
    RIGHT = 63
    TOTAL = 126
    """
    left_features = hand_to_features(left_hand)
    right_features = hand_to_features(right_hand)
    return left_features + right_features


# ============================================================
# DRAW HAND LANDMARKS
# ============================================================

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


# ============================================================
# FACIAL EXPRESSION DETECTION
# ============================================================

def get_eyebrow_raise_score(face_result):
    """
    Returns a 0-1 score for how raised the eyebrows are,
    using MediaPipe blendshapes. Higher = more raised.
    """
    if not face_result or not face_result.face_blendshapes:
        return 0.0

    blendshapes = face_result.face_blendshapes[0]

    target_names = {"browInnerUp", "browOuterUpLeft", "browOuterUpRight"}
    scores = [b.score for b in blendshapes if b.category_name in target_names]

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def get_nose_tip(face_result):
    """
    Returns (x, y) of the nose tip landmark (index 1), normalized 0-1,
    or None if no face detected.
    """
    if not face_result or not face_result.face_landmarks:
        return None

    nose = face_result.face_landmarks[0][1]
    return (nose.x, nose.y)


def detect_facial_marker(face_result, head_history):
    """
    Combines eyebrow raise + head movement pattern to decide the
    current facial/grammatical marker:

        "QUESTION" -> eyebrows raised
        "NEGATIVE" -> head shaking side to side (no)
        "POSITIVE" -> head nodding up and down (yes)
        None       -> neutral / no clear signal

    head_history: a deque of (x, y) nose positions, updated by caller
    every frame BEFORE calling this function.
    """
    eyebrow_score = get_eyebrow_raise_score(face_result)

    if eyebrow_score > EYEBROW_RAISE_THRESHOLD:
        return "QUESTION"

    if len(head_history) < head_history.maxlen:
        return None  # not enough data yet

    xs = [p[0] for p in head_history]
    ys = [p[1] for p in head_history]

    x_std = float(np.std(xs))
    y_std = float(np.std(ys))

    # Head shaking side-to-side (x moves more than y) -> NEGATIVE
    if x_std > HEAD_SHAKE_STD_THRESHOLD and x_std > y_std:
        return "NEGATIVE"

    # Head nodding up-down (y moves more than x) -> POSITIVE
    if y_std > HEAD_NOD_STD_THRESHOLD and y_std > x_std:
        return "POSITIVE"

    return None


# ============================================================
# TEXT TO SPEECH
# ============================================================

def speak_text(text):
    try:
        tts = gTTS(text=text, lang="en")
        tts.save("temp_speech.mp3")
        os.system("start temp_speech.mp3")
    except Exception as e:
        print(f"Speech generation error: {e}")


# ============================================================
# VOICE-TO-SIGN (offline, using Vosk)
# ============================================================

def listen_and_transcribe(result_holder, duration=4):
    """
    Records `duration` seconds of audio from the mic and transcribes
    it offline using Vosk. Puts the recognized text into result_holder
    (a single-element list, used so the background thread can hand the
    result back to the main thread).
    """
    if vosk_model is None:
        result_holder[0] = "(Voice-to-Sign not available - Vosk model missing)"
        return

    sample_rate = 16000
    recognizer = KaldiRecognizer(vosk_model, sample_rate)
    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_queue.put(bytes(indata))

    try:
        with sd.RawInputStream(
            samplerate=sample_rate, blocksize=8000, dtype="int16",
            channels=1, callback=callback
        ):
            start_time = time.time()
            while time.time() - start_time < duration:
                data = audio_queue.get()
                recognizer.AcceptWaveform(data)

        final_result = json.loads(recognizer.FinalResult())
        text = final_result.get("text", "").strip()

        result_holder[0] = text if text else "(couldn't catch that, try again)"

    except Exception as e:
        result_holder[0] = f"(Voice-to-Sign error: {e})"


# ============================================================
# GET TOP PREDICTIONS
# ============================================================

def get_top_predictions(model, input_data, top_n=3):
    """
    Random Forest:
    Top 3 predictions return pannum.
    """
    probabilities = model.predict_proba(input_data)[0]
    class_names = model.classes_

    top_indices = np.argsort(probabilities)[-top_n:][::-1]

    predictions = []
    for index in top_indices:
        label = str(class_names[index])
        confidence = float(probabilities[index]) * 100
        predictions.append((label, confidence))

    return predictions


# ============================================================
# CHECK AMBIGUOUS SIGN
# ============================================================

def is_ambiguous(predictions):
    if len(predictions) < 2:
        return False

    sign1, conf1 = predictions[0]
    sign2, conf2 = predictions[1]

    difference = abs(conf1 - conf2)
    if difference > AMBIGUITY_THRESHOLD:
        return False

    pair = {sign1.lower().strip(), sign2.lower().strip()}

    for group in AMBIGUOUS_SIGN_GROUPS:
        if pair.issubset(group):
            return True

    return False


# ============================================================
# DISAMBIGUATE SIGN
# ============================================================

def disambiguate_sign(predictions, sentence_words, facial_marker=None):
    """
    Context + facial marker use panni
    ambiguous sign decide pannum.
    """
    if not predictions:
        return None

    best_sign = predictions[0][0]

    if len(predictions) < 2:
        return best_sign

    sign1, conf1 = predictions[0]
    sign2, conf2 = predictions[1]

    s1 = sign1.lower().strip()
    s2 = sign2.lower().strip()

    difference = abs(conf1 - conf2)

    # Strong prediction
    if difference > AMBIGUITY_THRESHOLD:
        return best_sign

    context = " ".join(str(word) for word in sentence_words).lower()

    # ========================================================
    # CAN / CAN'T
    # ========================================================
    if {s1, s2} == {"can", "can't"}:
        if facial_marker == "NEGATIVE":
            return sign1 if s1 == "can't" else sign2
        if "not" in context or "never" in context:
            return sign1 if s1 == "can't" else sign2
        return sign1 if s1 == "can" else sign2

    # ========================================================
    # YES / NO
    # ========================================================
    if {s1, s2} == {"yes", "no"}:
        if facial_marker == "NEGATIVE":
            return sign1 if s1 == "no" else sign2
        if facial_marker == "POSITIVE":
            return sign1 if s1 == "yes" else sign2

    # ========================================================
    # GOOD / BAD
    # ========================================================
    if {s1, s2} == {"good", "bad"}:
        if facial_marker == "NEGATIVE":
            return sign1 if s1 == "bad" else sign2
        if facial_marker == "POSITIVE":
            return sign1 if s1 == "good" else sign2

    # ========================================================
    # COME / GO
    # ========================================================
    if {s1, s2} == {"come", "go"}:
        if "here" in context:
            return sign1 if s1 == "come" else sign2
        if "there" in context:
            return sign1 if s1 == "go" else sign2

    # ========================================================
    # QUESTION
    # ========================================================
    if facial_marker == "QUESTION":
        question_words = {"what", "where", "when", "why", "who", "how"}
        if s1 in question_words:
            return sign1
        if s2 in question_words:
            return sign2

    # ========================================================
    # DEFAULT
    # ========================================================
    return best_sign


# ============================================================
# GET WORD SUGGESTIONS
# ============================================================

def get_suggestions(sentence):
    """
    Looks up likely next words based on the last word (or last two
    words) of the current sentence. Returns a list of up to 3 words.
    """
    if not sentence:
        return []

    last_word = str(sentence[-1]).lower().strip()
    last_two = " ".join(str(w).lower().strip() for w in sentence[-2:])

    if last_two in NEXT_WORD_SUGGESTIONS:
        return NEXT_WORD_SUGGESTIONS[last_two][:3]

    if last_word in NEXT_WORD_SUGGESTIONS:
        return NEXT_WORD_SUGGESTIONS[last_word][:3]

    return []


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CAMERA
    # ========================================================
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera open panna mudiyala.")
        return

    # ========================================================
    # SENTENCE VARIABLES
    # ========================================================
    sentence = []
    current_word = ""
    hold_start_time = None
    last_added_word = ""

    # ========================================================
    # EMERGENCY VARIABLES
    # ========================================================
    emergency_mode = False
    emergency_message = ""

    # ========================================================
    # AMBIGUOUS VARIABLES
    # ========================================================
    top_predictions = []
    ambiguous_mode = False

    # ========================================================
    # FACIAL MARKER (NOW ACTIVE)
    # ========================================================
    facial_marker = None
    head_history = deque(maxlen=HEAD_MOVEMENT_HISTORY_LEN)

    # ========================================================
    # VOICE-TO-SIGN VARIABLES
    # ========================================================
    voice_to_sign_mode = False
    voice_to_sign_text = ""
    voice_to_sign_display_until = 0
    voice_result_holder = [None]

    # ========================================================
    # PREDICTIVE SUGGESTIONS
    # ========================================================
    current_suggestions = []

    # ========================================================
    # START
    # ========================================================
    print("=" * 60)
    print("       SIGN LANGUAGE RECOGNITION APP")
    print("=" * 60)
    print()
    print("App ready!")
    print("Sign pannunga, 1.5 sec hold pannina sentence la add aagum.")
    print()
    print("CONTROLS:")
    print("c = Clear sentence")
    print("v = Voice (speak sentence)")
    print("b = Backspace")
    print("e = Reset Emergency Mode")
    print("m = Voice-to-Sign (listen to mic)")
    print("1/2/3 = Pick a suggested next word")
    print("q = Quit")
    print()
    print("FEATURES:")
    print("Two-Hand Recognition = ON")
    print("126 Features = ON")
    print("Ambiguous Sign Disambiguation = ON")
    print(f"Facial Expression Detection = {'ON' if FACE_DETECTION_ENABLED else 'OFF (model missing)'}")
    print(f"Voice-to-Sign = {'ON' if VOICE_TO_SIGN_ENABLED else 'OFF (vosk model missing)'}")
    print("Predictive Word Suggestions = ON")
    print()
    print("Emergency Signs:")
    print("help      -> I need help")
    print("doctor    -> I need a doctor")
    print("emergency -> Emergency")
    print("family    -> Please call my family")
    print("ambulance -> Please call an ambulance")
    print()

    # ========================================================
    # CAMERA LOOP
    # ========================================================
    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            print("Camera frame read panna mudiyala.")
            break

        # MIRROR
        frame = cv2.flip(frame, 1)

        # RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # ====================================================
        # HAND DETECTION
        # ====================================================
        result = landmarker.detect(mp_image)

        prediction = None
        confidence = 0.0
        top_predictions = []
        ambiguous_mode = False

        if result.hand_landmarks:
            draw_hand_landmarks(frame, result.hand_landmarks)

            left_hand, right_hand = get_hand_landmarks(result)
            features = create_two_hand_features(left_hand, right_hand)

            if len(features) != 126:
                print(f"Feature Error: {len(features)} features")
                continue

            input_data = np.array(features).reshape(1, -1)

            top_predictions = get_top_predictions(model, input_data, top_n=3)

            pred = top_predictions[0][0]
            conf = top_predictions[0][1]

            ambiguous_mode = is_ambiguous(top_predictions)

            if conf >= CONFIDENCE_THRESHOLD:
                if ambiguous_mode:
                    prediction = disambiguate_sign(top_predictions, sentence, facial_marker)
                    confidence = conf
                    for sign, sign_conf in top_predictions:
                        if sign == prediction:
                            confidence = sign_conf
                            break
                else:
                    prediction = pred
                    confidence = conf

        # ====================================================
        # FACE DETECTION (facial marker: QUESTION / NEGATIVE / POSITIVE)
        # ====================================================
        if FACE_DETECTION_ENABLED:
            face_result = face_landmarker.detect(mp_image)

            nose_pos = get_nose_tip(face_result)
            if nose_pos is not None:
                head_history.append(nose_pos)

            facial_marker = detect_facial_marker(face_result, head_history)
        else:
            facial_marker = None

        # ====================================================
        # HOLD TO CONFIRM
        # ====================================================
        if prediction:

            if prediction == current_word:
                elapsed = (time.time() - hold_start_time) if hold_start_time is not None else 0

                if elapsed >= HOLD_DURATION and prediction != last_added_word:
                    sentence.append(prediction)
                    last_added_word = prediction
                    hold_start_time = time.time()

                    # Refresh predictive suggestions for the new sentence state
                    current_suggestions = get_suggestions(sentence)

                    # EMERGENCY CHECK
                    prediction_key = str(prediction).lower().strip()
                    if prediction_key in EMERGENCY_SIGNS:
                        emergency_mode = True
                        emergency_message = EMERGENCY_SIGNS[prediction_key]

                        print()
                        print("=" * 60)
                        print("🚨 EMERGENCY MODE ACTIVATED!")
                        print("=" * 60)
                        print(f"Emergency Message: {emergency_message}")
                        print("=" * 60)

                        threading.Thread(target=speak_text, args=(emergency_message,)).start()

                    # AMBIGUOUS INFORMATION
                    if ambiguous_mode:
                        print()
                        print("AMBIGUOUS SIGN DETECTED")
                        print(f"Facial marker at the time: {facial_marker}")
                        print("Top Predictions:")
                        for sign, conf in top_predictions:
                            print(f"  {sign}: {conf:.1f}%")
                        print(f"Final Decision: {prediction}")
                        print()

            else:
                current_word = prediction
                hold_start_time = time.time()
                last_added_word = ""

        else:
            current_word = ""
            hold_start_time = None
            last_added_word = ""

        # ====================================================
        # VOICE-TO-SIGN: check if background listening finished
        # ====================================================
        if voice_to_sign_mode and voice_result_holder[0] is not None:
            voice_to_sign_text = voice_result_holder[0]
            voice_to_sign_display_until = time.time() + 6  # show for 6 seconds
            voice_to_sign_mode = False
            voice_result_holder[0] = None

        # ====================================================
        # UI
        # ====================================================
        h, w, _ = frame.shape

        # TOP BAR
        cv2.rectangle(frame, (0, 0), (w, 105), (0, 0, 0), -1)

        if prediction:
            hold_progress = min((time.time() - hold_start_time) / HOLD_DURATION, 1.0) if hold_start_time else 0

            cv2.putText(frame, f"Detecting: {prediction} ({confidence:.0f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            if ambiguous_mode:
                cv2.putText(frame, "AMBIGUOUS - CONTEXT CHECK", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                if len(top_predictions) >= 2:
                    p1, c1 = top_predictions[0]
                    p2, c2 = top_predictions[1]
                    cv2.putText(frame, f"{p1}: {c1:.0f}% | {p2}: {c2:.0f}%", (10, 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.rectangle(frame, (10, 88), (310, 101), (100, 100, 100), 1)
            bar_width = int(300 * hold_progress)
            cv2.rectangle(frame, (10, 88), (10 + bar_width, 101), (0, 255, 255), -1)
        else:
            cv2.putText(frame, "Show a sign...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        # FACIAL MARKER
        if facial_marker:
            marker_color = {
                "QUESTION": (0, 255, 255),
                "NEGATIVE": (0, 0, 255),
                "POSITIVE": (0, 255, 0),
            }.get(facial_marker, (255, 255, 0))

            marker_label = {
                "QUESTION": "🤨 QUESTION (eyebrows up)",
                "NEGATIVE": "🙅 NEGATIVE (head shake)",
                "POSITIVE": "🙆 POSITIVE (head nod)",
            }.get(facial_marker, facial_marker)

            cv2.putText(frame, marker_label, (w - 420, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, marker_color, 2)

        # EMERGENCY UI
        if emergency_mode:
            cv2.rectangle(frame, (0, 110), (w, 195), (0, 0, 255), -1)
            cv2.putText(frame, "!!! EMERGENCY MODE !!!", (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
            cv2.putText(frame, emergency_message, (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # PREDICTIVE SUGGESTIONS BAR
        if current_suggestions and not emergency_mode:
            sugg_y = 215
            cv2.rectangle(frame, (0, sugg_y - 30), (w, sugg_y + 10), (40, 40, 40), -1)
            suggestion_text = " | ".join(
                f"[{i+1}] {word}" for i, word in enumerate(current_suggestions)
            )
            cv2.putText(frame, f"Suggestions: {suggestion_text}", (10, sugg_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # VOICE-TO-SIGN UI
        if voice_to_sign_mode:
            cv2.rectangle(frame, (0, h - 140), (w, h - 70), (80, 0, 80), -1)
            cv2.putText(frame, "🎤 Listening... speak now", (20, h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif time.time() < voice_to_sign_display_until:
            cv2.rectangle(frame, (0, h - 140), (w, h - 70), (80, 0, 80), -1)
            cv2.putText(frame, f"Heard: {voice_to_sign_text}", (20, h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # SENTENCE BAR
        cv2.rectangle(frame, (0, h - 70), (w, h), (30, 30, 30), -1)
        sentence_text = " ".join(sentence) if sentence else "(sentence empty)"
        cv2.putText(frame, sentence_text, (10, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # CAMERA WINDOW
        cv2.imshow("Sign Language Recognition App", frame)

        # ====================================================
        # KEYBOARD
        # ====================================================
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            sentence = []
            current_suggestions = []
            print("Sentence cleared.")

        elif key == ord("b"):
            if sentence:
                removed = sentence.pop()
                current_suggestions = get_suggestions(sentence)
                print(f"Removed: {removed}")

        elif key == ord("v"):
            if sentence:
                full_text = " ".join(sentence)
                print(f"Speaking: {full_text}")
                threading.Thread(target=speak_text, args=(full_text,)).start()

        elif key == ord("e"):
            emergency_mode = False
            emergency_message = ""
            print("Emergency Mode reset.")

        elif key == ord("m"):
            if VOICE_TO_SIGN_ENABLED and not voice_to_sign_mode:
                voice_to_sign_mode = True
                voice_result_holder[0] = None
                print("Voice-to-Sign: listening for 4 seconds...")
                threading.Thread(
                    target=listen_and_transcribe,
                    args=(voice_result_holder, 4)
                ).start()
            elif not VOICE_TO_SIGN_ENABLED:
                print("Voice-to-Sign not available - vosk-model folder missing.")

        elif key in (ord("1"), ord("2"), ord("3")):
            idx = key - ord("1")
            if idx < len(current_suggestions):
                chosen_word = current_suggestions[idx]
                sentence.append(chosen_word)
                print(f"Added via suggestion: {chosen_word}")
                current_suggestions = get_suggestions(sentence)

    # ========================================================
    # RELEASE
    # ========================================================
    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# START APP
# ============================================================

if __name__ == "__main__":
    main()
