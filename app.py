```python
"""
Sign Language Recognition - Full App
========================================
Real-time sign detection + sentence building + text-to-speech
+ Emergency Mode

CONTROLS:
- Sign ah 1.5 seconds hold pannunga -> sentence la add aagum
- 'c' key -> sentence clear pannum
- 'v' key -> sentence ah voice ah sollum
- 'b' key -> last word remove pannum
- 'e' key -> Emergency Mode reset pannum
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


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "model/sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"

CONFIDENCE_THRESHOLD = 70.0
HOLD_DURATION = 1.5


# ============================================================
# EMERGENCY SIGNS
# ============================================================
# IMPORTANT:
# Inga irukkura names unga model labels-oda
# EXACT SAME spelling ah irukkanum.

EMERGENCY_SIGNS = {
    "help": "I need help",
    "doctor": "I need a doctor",
    "emergency": "Emergency",
    "family": "Please call my family",
    "ambulance": "Please call an ambulance"
}


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

with open(MODEL_PATH, 'rb') as f:
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
            lang='en'
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


    print("=" * 60)
    print("       SIGN LANGUAGE RECOGNITION APP")
    print("=" * 60)

    print()
    print("App ready!")
    print("Sign pannunga, 1.5 sec hold pannina sentence la add aagum.")
    print()

    print("CONTROLS:")
    print("c = Clear sentence")
    print("v = Voice")
    print("b = Backspace")
    print("e = Reset Emergency Mode")
    print("q = Quit")

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


            # Predict sign
            pred = model.predict(
                input_data
            )[0]


            # Probability
            probabilities = model.predict_proba(
                input_data
            )[0]


            conf = max(probabilities) * 100


            # Confidence check
            if conf >= CONFIDENCE_THRESHOLD:

                prediction = pred

                confidence = conf


        # ====================================================
        # HOLD-TO-CONFIRM LOGIC
        # ====================================================

        if prediction:

            # Same sign continues
            if prediction == current_word:

                elapsed = (
                    time.time()
                    - hold_start_time
                )


                # Sign held for 1.5 seconds
                if (
                    elapsed >= HOLD_DURATION
                    and prediction != last_added_word
                ):

                    # Add sign to sentence
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
                        print("🚨 EMERGENCY MODE ACTIVATED!")
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


            # New sign
            else:

                current_word = prediction

                hold_start_time = time.time()

                last_added_word = ""


        # No sign detected
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
            (w, 70),
            (0, 0, 0),
            -1
        )


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


            # Progress bar background
            cv2.rectangle(
                frame,
                (10, 45),
                (310, 60),
                (100, 100, 100),
                1
            )


            # Progress
            bar_width = int(
                300 * hold_progress
            )


            cv2.rectangle(
                frame,
                (10, 45),
                (
                    10 + bar_width,
                    60
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
        # EMERGENCY ALERT UI
        # ====================================================

        if emergency_mode:

            # Red alert box
            cv2.rectangle(
                frame,
                (0, 70),
                (w, 155),
                (0, 0, 255),
                -1
            )


            # Emergency title
            cv2.putText(
                frame,
                "!!! EMERGENCY MODE !!!",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                3
            )


            # Emergency message
            cv2.putText(
                frame,
                emergency_message,
                (20, 140),
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
            (0, h - 60),
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
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW CAMERA WINDOW
        # ====================================================

        cv2.imshow(
            'Sign Language Recognition App',
            frame
        )


        # ====================================================
        # KEYBOARD CONTROLS
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # Quit
        if key == ord('q'):

            break


        # Clear sentence
        elif key == ord('c'):

            sentence = []

            print(
                "Sentence cleared."
            )


        # Backspace
        elif key == ord('b'):

            if sentence:

                removed = sentence.pop()

                print(
                    f"Removed: {removed}"
                )


        # Voice
        elif key == ord('v'):

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

        elif key == ord('e'):

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
```
