"""
Sign Language Recognition - Full App
========================================
Real-time sign detection
+ Sentence building
+ Text-to-speech
+ Emergency Mode
+ Ambiguous Sign Disambiguation

CONTROLS:
- Sign ah 1.5 seconds hold pannunga -> sentence la add aagum
- 'c' key -> sentence clear pannum
- 'v' key -> sentence ah voice ah sollum
- 'b' key -> last word remove pannum
- 'e' key -> Emergency Mode reset pannum
- 'q' key -> exit pannum

NEW:
- Ambiguous signs -> top predictions compare pannum
- Previous sentence context use pannum
- Similar confidence predictions irundha context-based decision
- Future facial expression integration-ku ready
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


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "model/sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"

CONFIDENCE_THRESHOLD = 70.0
HOLD_DURATION = 1.5

# Ambiguous sign threshold
# Top 2 predictions difference <= this value
# na ambiguous sign-nu consider pannuvom
AMBIGUITY_THRESHOLD = 15.0


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
# Unga trained model labels different-a irundha
# inga labels-ai unga model labels-ku match pannunga.
#
# Example:
# CAN and CAN'T visually similar signs na,
# ["can", "can't"] nu group create pannalam.
#
# YES and NO etc. similarly add pannalam.
#

AMBIGUOUS_SIGN_GROUPS = [
    {"can", "can't"},
    {"yes", "no"},
    {"good", "bad"},
    {"come", "go"},
    {"here", "there"},
    {"this", "that"},
]


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: '{MODEL_PATH}' kedaikala.")
    print("Munnadi 'train_model.py' run pannunga.")
    exit(1)

if not os.path.exists(HAND_MODEL_PATH):
    print(f"ERROR: '{HAND_MODEL_PATH}' kedaikala.")
    print("hand_landmarker.task file project folder-la irukkanum.")
    exit(1)


# ============================================================
# LOAD SIGN RECOGNITION MODEL
# ============================================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

base_options = mp_python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    running_mode=vision.RunningMode.IMAGE
)

landmarker = vision.HandLandmarker.create_from_options(options)


# ============================================================
# NORMALIZE LANDMARKS
# ============================================================

def normalize_landmarks(landmarks):

    base_x = landmarks[0][0]
    base_y = landmarks[0][1]

    normalized = []

    for x, y, z in landmarks:
        normalized.extend([
            x - base_x,
            y - base_y,
            z
        ])

    return normalized


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

        points = [
            (int(lm.x * w), int(lm.y * h))
            for lm in hand_landmarks
        ]

        # Draw connections
        for start_idx, end_idx in connections:

            cv2.line(
                frame,
                points[start_idx],
                points[end_idx],
                (0, 255, 0),
                2
            )

        # Draw points
        for point in points:

            cv2.circle(
                frame,
                point,
                4,
                (0, 0, 255),
                -1
            )


# ============================================================
# TEXT TO SPEECH
# ============================================================

def speak_text(text):

    """
    Background thread-la audio generate pannum.
    App freeze aagama irukka thread use pannrom.
    """

    try:

        tts = gTTS(
            text=text,
            lang="en"
        )

        tts.save("temp_speech.mp3")

        os.system(
            "start temp_speech.mp3"
        )

    except Exception as e:

        print(
            f"Speech generation error: {e}"
        )


# ============================================================
# GET TOP PREDICTIONS
# ============================================================

def get_top_predictions(model, input_data, top_n=3):

    """
    Model oda top N predictions return pannum.

    Example:

    [
        ("CAN", 72.5),
        ("CAN'T", 69.8),
        ("GO", 20.1)
    ]
    """

    probabilities = model.predict_proba(input_data)[0]

    class_names = model.classes_

    top_indices = np.argsort(
        probabilities
    )[-top_n:][::-1]

    predictions = []

    for index in top_indices:

        label = str(class_names[index])

        confidence = (
            float(probabilities[index]) * 100
        )

        predictions.append(
            (label, confidence)
        )

    return predictions


# ============================================================
# CHECK AMBIGUITY
# ============================================================

def is_ambiguous(predictions):

    """
    Top 2 predictions confidence close-a irundha
    ambiguous sign-nu consider pannuvom.
    """

    if len(predictions) < 2:
        return False

    sign1, conf1 = predictions[0]
    sign2, conf2 = predictions[1]

    difference = abs(
        conf1 - conf2
    )

    if difference <= AMBIGUITY_THRESHOLD:

        pair = {
            sign1.lower().strip(),
            sign2.lower().strip()
        }

        for group in AMBIGUOUS_SIGN_GROUPS:

            if pair.issubset(group):
                return True

    return False


# ============================================================
# CONTEXT-BASED DISAMBIGUATION
# ============================================================

def disambiguate_sign(
    predictions,
    sentence_words,
    facial_marker=None
):

    """
    Ambiguous signs-ku context based decision.

    predictions:
        [("CAN", 72), ("CAN'T", 69)]

    sentence_words:
        ["I", "go"]

    facial_marker:
        None
        "QUESTION"
        "NEGATIVE"
        "POSITIVE"
    """

    if not predictions:
        return None

    # Highest confidence prediction
    best_sign = predictions[0][0]

    if len(predictions) < 2:
        return best_sign

    sign1, conf1 = predictions[0]
    sign2, conf2 = predictions[1]

    sign1_lower = sign1.lower().strip()
    sign2_lower = sign2.lower().strip()

    # Confidence difference
    difference = abs(
        conf1 - conf2
    )

    # Not ambiguous
    if difference > AMBIGUITY_THRESHOLD:
        return best_sign

    # Previous sentence
    context = " ".join(
        str(word) for word in sentence_words
    ).lower()

    # ========================================================
    # CAN / CAN'T
    # ========================================================

    if {
        sign1_lower,
        sign2_lower
    } == {"can", "can't"}:

        if facial_marker == "NEGATIVE":
            return (
                sign1
                if sign1_lower == "can't"
                else sign2
            )

        if "not" in context:
            return (
                sign1
                if sign1_lower == "can't"
                else sign2
            )

        if "never" in context:
            return (
                sign1
                if sign1_lower == "can't"
                else sign2
            )

        return (
            sign1
            if sign1_lower == "can"
            else sign2
        )

    # ========================================================
    # YES / NO
    # ========================================================

    if {
        sign1_lower,
        sign2_lower
    } == {"yes", "no"}:

        if facial_marker == "NEGATIVE":

            return (
                sign1
                if sign1_lower == "no"
                else sign2
            )

        if facial_marker == "POSITIVE":

            return (
                sign1
                if sign1_lower == "yes"
                else sign2
            )

    # ========================================================
    # GOOD / BAD
    # ========================================================

    if {
        sign1_lower,
        sign2_lower
    } == {"good", "bad"}:

        if facial_marker == "NEGATIVE":

            return (
                sign1
                if sign1_lower == "bad"
                else sign2
            )

        if facial_marker == "POSITIVE":

            return (
                sign1
                if sign1_lower == "good"
                else sign2
            )

    # ========================================================
    # COME / GO
    # ========================================================

    if {
        sign1_lower,
        sign2_lower
    } == {"come", "go"}:

        if "here" in context:

            return (
                sign1
                if sign1_lower == "come"
                else sign2
            )

        if "there" in context:

            return (
                sign1
                if sign1_lower == "go"
                else sign2
            )

    # ========================================================
    # QUESTION MARKER
    # ========================================================

    if facial_marker == "QUESTION":

        question_words = {
            "what",
            "where",
            "when",
            "why",
            "who",
            "how"
        }

        if sign1_lower in question_words:
            return sign1

        if sign2_lower in question_words:
            return sign2

    # ========================================================
    # DEFAULT
    # ========================================================

    return best_sign


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # Camera
    cap = cv2.VideoCapture(0)

    # Sentence variables
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
    # AMBIGUOUS SIGN VARIABLES
    # ========================================================

    top_predictions = []

    ambiguous_mode = False

    # Facial marker placeholder
    #
    # Future MediaPipe FaceLandmarker integration:
    #
    # "QUESTION"
    # "NEGATIVE"
    # "POSITIVE"
    #
    # Ippo None.
    facial_marker = None

    # ========================================================
    # START MESSAGE
    # ========================================================

    print("=" * 60)
    print("       SIGN LANGUAGE RECOGNITION APP")
    print("=" * 60)

    print()
    print("App ready!")
    print(
        "Sign pannunga, 1.5 sec hold pannina "
        "sentence la add aagum."
    )
    print()

    print("CONTROLS:")
    print("c = Clear sentence")
    print("v = Voice")
    print("b = Backspace")
    print("e = Reset Emergency Mode")
    print("q = Quit")

    print()

    print("NEW FEATURE:")
    print(
        "Ambiguous Sign Disambiguation enabled."
    )

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

            print(
                "Camera access panna mudiyala."
            )

            break

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        # ====================================================
        # CONVERT FRAME FOR MEDIAPIPE
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # ====================================================
        # HAND DETECTION
        # ====================================================

        result = landmarker.detect(
            mp_image
        )

        prediction = None

        confidence = 0.0

        top_predictions = []

        ambiguous_mode = False

        # ====================================================
        # SIGN PREDICTION
        # ====================================================

        if result.hand_landmarks:

            # Draw landmarks
            draw_hand_landmarks(
                frame,
                result.hand_landmarks
            )

            landmark_list = [
                (lm.x, lm.y, lm.z)
                for lm in result.hand_landmarks[0]
            ]

            # Normalize
            normalized = normalize_landmarks(
                landmark_list
            )

            # Convert to numpy
            input_data = np.array(
                normalized
            ).reshape(
                1,
                -1
            )

            # =================================================
            # TOP PREDICTIONS
            # =================================================

            top_predictions = get_top_predictions(
                model,
                input_data,
                top_n=3
            )

            # Highest confidence
            pred = top_predictions[0][0]

            conf = top_predictions[0][1]

            # =================================================
            # AMBIGUITY CHECK
            # =================================================

            ambiguous_mode = is_ambiguous(
                top_predictions
            )

            # =================================================
            # CONFIDENCE CHECK
            # =================================================

            if conf >= CONFIDENCE_THRESHOLD:

                # If ambiguous:
                # context + facial marker use pannum
                if ambiguous_mode:

                    prediction = disambiguate_sign(
                        top_predictions,
                        sentence,
                        facial_marker
                    )

                    # Highest selected confidence
                    selected_conf = conf

                    for sign, sign_conf in top_predictions:

                        if sign == prediction:
                            selected_conf = sign_conf
                            break

                    confidence = selected_conf

                else:

                    prediction = pred

                    confidence = conf

        # ====================================================
        # HOLD-TO-CONFIRM LOGIC
        # ====================================================

        if prediction:

            # Same sign continues
            if prediction == current_word:

                if hold_start_time is not None:

                    elapsed = (
                        time.time()
                        - hold_start_time
                    )

                else:

                    elapsed = 0

                # Sign held for 1.5 seconds
                if (
                    elapsed >= HOLD_DURATION
                    and prediction != last_added_word
                ):

                    # ========================================
                    # ADD SIGN TO SENTENCE
                    # ========================================

                    sentence.append(
                        prediction
                    )

                    last_added_word = prediction

                    hold_start_time = time.time()

                    # ========================================
                    # EMERGENCY MODE CHECK
                    # ========================================

                    prediction_key = str(
                        prediction
                    ).lower().strip()

                    if prediction_key in EMERGENCY_SIGNS:

                        # Activate Emergency Mode
                        emergency_mode = True

                        # Get emergency message
                        emergency_message = (
                            EMERGENCY_SIGNS[
                                prediction_key
                            ]
                        )

                        print()
                        print("=" * 60)
                        print(
                            "🚨 EMERGENCY MODE ACTIVATED!"
                        )
                        print("=" * 60)

                        print(
                            f"Emergency Message: "
                            f"{emergency_message}"
                        )

                        print("=" * 60)

                        # Speak emergency message
                        threading.Thread(
                            target=speak_text,
                            args=(emergency_message,)
                        ).start()

                    # ========================================
                    # AMBIGUOUS SIGN INFORMATION
                    # ========================================

                    if ambiguous_mode:

                        print()
                        print(
                            "Ambiguous Sign Detected"
                        )

                        print(
                            "Top Predictions:"
                        )

                        for sign, conf in top_predictions:

                            print(
                                f"  {sign}: "
                                f"{conf:.1f}%"
                            )

                        print(
                            f"Context Decision: "
                            f"{prediction}"
                        )

                        print()

            # New sign
            else:

                current_word = prediction

                hold_start_time = time.time()

                last_added_word = ""

        # ====================================================
        # NO SIGN DETECTED
        # ====================================================

        else:

            current_word = ""

            hold_start_time = None

            last_added_word = ""

        # ====================================================
        # UI DRAWING
        # ====================================================

        h, w, _ = frame.shape

        # ====================================================
        # TOP BAR
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (w, 100),
            (0, 0, 0),
            -1
        )

        # ====================================================
        # PREDICTION DISPLAY
        # ====================================================

        if prediction:

            # Hold progress
            if hold_start_time:

                hold_progress = min(
                    (
                        time.time()
                        - hold_start_time
                    )
                    / HOLD_DURATION,
                    1.0
                )

            else:

                hold_progress = 0

            # Prediction text
            cv2.putText(
                frame,
                f"Detecting: {prediction} "
                f"({confidence:.0f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # =================================================
            # AMBIGUOUS DISPLAY
            # =================================================

            if ambiguous_mode:

                cv2.putText(
                    frame,
                    "AMBIGUOUS SIGN - CONTEXT CHECK",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                # Show top 2 predictions
                if len(top_predictions) >= 2:

                    p1, c1 = top_predictions[0]
                    p2, c2 = top_predictions[1]

                    cv2.putText(
                        frame,
                        f"{p1}: {c1:.0f}% | "
                        f"{p2}: {c2:.0f}%",
                        (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

            # Progress bar background
            cv2.rectangle(
                frame,
                (10, 85),
                (310, 98),
                (100, 100, 100),
                1
            )

            # Progress
            bar_width = int(
                300 * hold_progress
            )

            cv2.rectangle(
                frame,
                (10, 85),
                (
                    10 + bar_width,
                    98
                ),
                (0, 255, 255),
                -1
            )

        else:

            cv2.putText(
                frame,
                "Show a sign...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (150, 150, 150),
                2
            )

        # ====================================================
        # FACIAL MARKER DISPLAY
        # ====================================================

        if facial_marker:

            cv2.putText(
                frame,
                f"Face Marker: {facial_marker}",
                (w - 300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

        # ====================================================
        # EMERGENCY ALERT UI
        # ====================================================

        if emergency_mode:

            # Red alert box
            cv2.rectangle(
                frame,
                (0, 105),
                (w, 190),
                (0, 0, 255),
                -1
            )

            # Emergency title
            cv2.putText(
                frame,
                "!!! EMERGENCY MODE !!!",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                3
            )

            # Emergency message
            cv2.putText(
                frame,
                emergency_message,
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        # ====================================================
        # BOTTOM SENTENCE BAR
        # ====================================================

        cv2.rectangle(
            frame,
            (0, h - 70),
            (w, h),
            (30, 30, 30),
            -1
        )

        sentence_text = (
            " ".join(sentence)
            if sentence
            else "(sentence empty)"
        )

        cv2.putText(
            frame,
            sentence_text,
            (10, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ====================================================
        # SHOW CAMERA WINDOW
        # ====================================================

        cv2.imshow(
            "Sign Language Recognition App",
            frame
        )

        # ====================================================
        # KEYBOARD CONTROLS
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q"):

            break

        # ====================================================
        # CLEAR SENTENCE
        # ====================================================

        elif key == ord("c"):

            sentence = []

            print(
                "Sentence cleared."
            )

        # ====================================================
        # BACKSPACE
        # ====================================================

        elif key == ord("b"):

            if sentence:

                removed = sentence.pop()

                print(
                    f"Removed: {removed}"
                )

        # ====================================================
        # VOICE
        # ====================================================

        elif key == ord("v"):

            if sentence:

                full_text = " ".join(
                    sentence
                )

                print(
                    f"Speaking: {full_text}"
                )

                threading.Thread(
                    target=speak_text,
                    args=(full_text,)
                ).start()

        # ====================================================
        # RESET EMERGENCY MODE
        # ====================================================

        elif key == ord("e"):

            emergency_mode = False

            emergency_message = ""

            print(
                "Emergency Mode reset."
            )

    # ========================================================
    # RELEASE CAMERA
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# START APP
# ============================================================

if __name__ == "__main__":

    main()