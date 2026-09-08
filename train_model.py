"""
Sign Language Recognition - Full App
========================================
Two-Hand Sign Recognition
+ Sentence Building
+ Text-to-Speech
+ Emergency Mode
+ Ambiguous Sign Disambiguation

MODEL:
- sign_data_two_hands.csv
- 126 features
- 21 landmarks x 3 coordinates x 2 hands

CONTROLS:
- Sign ah 1.5 seconds hold pannunga -> sentence la add aagum
- 'c' -> Clear sentence
- 'v' -> Voice
- 'b' -> Last word remove
- 'e' -> Emergency reset
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
from gtts import gTTS
import threading


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "model/sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"

CONFIDENCE_THRESHOLD = 70.0
HOLD_DURATION = 1.5

# Difference between top two predictions.
# If difference is <= 15%, we consider it ambiguous.
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


# ============================================================
# LOAD MODEL
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

    # IMPORTANT:
    # Model is trained using TWO hands
    num_hands=2,

    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4,

    running_mode=vision.RunningMode.IMAGE
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


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

    for i, hand_landmarks in enumerate(
        result.hand_landmarks
    ):

        # MediaPipe handedness
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

    features = []

    if hand_landmarks is None:

        return [0.0] * 63

    # Wrist as base point
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y

    for lm in hand_landmarks:

        features.extend([
            lm.x - base_x,
            lm.y - base_y,
            lm.z
        ])

    return features


# ============================================================
# CREATE 126 FEATURES
# ============================================================

def create_two_hand_features(
    left_hand,
    right_hand
):

    """
    LEFT  = 63
    RIGHT = 63

    TOTAL = 126
    """

    left_features = hand_to_features(
        left_hand
    )

    right_features = hand_to_features(
        right_hand
    )

    features = (
        left_features
        + right_features
    )

    return features


# ============================================================
# DRAW HAND LANDMARKS
# ============================================================

def draw_hand_landmarks(
    frame,
    hand_landmarks_list
):

    h, w, _ = frame.shape

    connections = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (0, 17)
    ]

    for hand_landmarks in hand_landmarks_list:

        points = [
            (
                int(lm.x * w),
                int(lm.y * h)
            )
            for lm in hand_landmarks
        ]

        # Connections
        for start_idx, end_idx in connections:

            cv2.line(
                frame,
                points[start_idx],
                points[end_idx],
                (0, 255, 0),
                2
            )

        # Points
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

    try:

        tts = gTTS(
            text=text,
            lang="en"
        )

        tts.save(
            "temp_speech.mp3"
        )

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

def get_top_predictions(
    model,
    input_data,
    top_n=3
):

    """
    Random Forest:
    Top 3 predictions return pannum.
    """

    probabilities = model.predict_proba(
        input_data
    )[0]

    class_names = model.classes_

    top_indices = np.argsort(
        probabilities
    )[-top_n:][::-1]

    predictions = []

    for index in top_indices:

        label = str(
            class_names[index]
        )

        confidence = (
            float(probabilities[index])
            * 100
        )

        predictions.append(
            (
                label,
                confidence
            )
        )

    return predictions


# ============================================================
# CHECK AMBIGUOUS SIGN
# ============================================================

def is_ambiguous(
    predictions
):

    if len(predictions) < 2:

        return False

    sign1, conf1 = predictions[0]
    sign2, conf2 = predictions[1]

    difference = abs(
        conf1 - conf2
    )

    if difference > AMBIGUITY_THRESHOLD:

        return False

    pair = {
        sign1.lower().strip(),
        sign2.lower().strip()
    }

    for group in AMBIGUOUS_SIGN_GROUPS:

        if pair.issubset(group):

            return True

    return False


# ============================================================
# DISAMBIGUATE SIGN
# ============================================================

def disambiguate_sign(
    predictions,
    sentence_words,
    facial_marker=None
):

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

    difference = abs(
        conf1 - conf2
    )

    # Strong prediction
    if difference > AMBIGUITY_THRESHOLD:

        return best_sign

    context = " ".join(
        str(word)
        for word in sentence_words
    ).lower()


    # ========================================================
    # CAN / CAN'T
    # ========================================================

    if {s1, s2} == {"can", "can't"}:

        if facial_marker == "NEGATIVE":

            if s1 == "can't":
                return sign1
            else:
                return sign2

        if "not" in context:

            if s1 == "can't":
                return sign1
            else:
                return sign2

        if "never" in context:

            if s1 == "can't":
                return sign1
            else:
                return sign2

        if s1 == "can":
            return sign1
        else:
            return sign2


    # ========================================================
    # YES / NO
    # ========================================================

    if {s1, s2} == {"yes", "no"}:

        if facial_marker == "NEGATIVE":

            if s1 == "no":
                return sign1
            else:
                return sign2

        if facial_marker == "POSITIVE":

            if s1 == "yes":
                return sign1
            else:
                return sign2


    # ========================================================
    # GOOD / BAD
    # ========================================================

    if {s1, s2} == {"good", "bad"}:

        if facial_marker == "NEGATIVE":

            if s1 == "bad":
                return sign1
            else:
                return sign2

        if facial_marker == "POSITIVE":

            if s1 == "good":
                return sign1
            else:
                return sign2


    # ========================================================
    # COME / GO
    # ========================================================

    if {s1, s2} == {"come", "go"}:

        if "here" in context:

            if s1 == "come":
                return sign1
            else:
                return sign2

        if "there" in context:

            if s1 == "go":
                return sign1
            else:
                return sign2


    # ========================================================
    # QUESTION
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

        if s1 in question_words:

            return sign1

        if s2 in question_words:

            return sign2


    # ========================================================
    # DEFAULT
    # ========================================================

    return best_sign


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "Camera open panna mudiyala."
        )

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
    # FACIAL MARKER
    # ========================================================
    #
    # Future FaceLandmarker integration:
    #
    # QUESTION
    # NEGATIVE
    # POSITIVE
    #

    facial_marker = None


    # ========================================================
    # START
    # ========================================================

    print(
        "=" * 60
    )

    print(
        "       SIGN LANGUAGE RECOGNITION APP"
    )

    print(
        "=" * 60
    )

    print()

    print(
        "App ready!"
    )

    print(
        "Sign pannunga, "
        "1.5 sec hold pannina "
        "sentence la add aagum."
    )

    print()

    print(
        "CONTROLS:"
    )

    print(
        "c = Clear sentence"
    )

    print(
        "v = Voice"
    )

    print(
        "b = Backspace"
    )

    print(
        "e = Reset Emergency Mode"
    )

    print(
        "q = Quit"
    )

    print()

    print(
        "FEATURES:"
    )

    print(
        "Two-Hand Recognition = ON"
    )

    print(
        "126 Features = ON"
    )

    print(
        "Ambiguous Sign Disambiguation = ON"
    )

    print()

    print(
        "Emergency Signs:"
    )

    print(
        "help      -> I need help"
    )

    print(
        "doctor    -> I need a doctor"
    )

    print(
        "emergency -> Emergency"
    )

    print(
        "family    -> Please call my family"
    )

    print(
        "ambulance -> Please call an ambulance"
    )

    print()


    # ========================================================
    # CAMERA LOOP
    # ========================================================

    while cap.isOpened():

        success, frame = cap.read()

        if not success:

            print(
                "Camera frame read panna mudiyala."
            )

            break


        # ====================================================
        # MIRROR
        # ====================================================

        frame = cv2.flip(
            frame,
            1
        )


        # ====================================================
        # RGB
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
        # TWO HAND PROCESSING
        # ====================================================

        if result.hand_landmarks:

            # Draw detected hands
            draw_hand_landmarks(
                frame,
                result.hand_landmarks
            )


            # Get left and right
            left_hand, right_hand = (
                get_hand_landmarks(result)
            )


            # =================================================
            # CREATE 126 FEATURES
            # =================================================

            features = create_two_hand_features(
                left_hand,
                right_hand
            )


            # Safety check
            if len(features) != 126:

                print(
                    f"Feature Error: "
                    f"{len(features)} features"
                )

                continue


            # =================================================
            # MODEL INPUT
            # =================================================

            input_data = np.array(
                features
            ).reshape(
                1,
                -1
            )


            # =================================================
            # TOP PREDICTIONS
            # =================================================

            top_predictions = (
                get_top_predictions(
                    model,
                    input_data,
                    top_n=3
                )
            )


            # Highest prediction
            pred = top_predictions[0][0]

            conf = top_predictions[0][1]


            # =================================================
            # AMBIGUOUS CHECK
            # =================================================

            ambiguous_mode = is_ambiguous(
                top_predictions
            )


            # =================================================
            # CONFIDENCE CHECK
            # =================================================

            if conf >= CONFIDENCE_THRESHOLD:

                if ambiguous_mode:

                    prediction = (
                        disambiguate_sign(
                            top_predictions,
                            sentence,
                            facial_marker
                        )
                    )

                    confidence = conf

                    # Selected sign confidence
                    for sign, sign_conf in top_predictions:

                        if sign == prediction:

                            confidence = sign_conf

                            break

                else:

                    prediction = pred

                    confidence = conf


        # ====================================================
        # HOLD TO CONFIRM
        # ====================================================

        if prediction:

            # Same sign
            if prediction == current_word:

                if hold_start_time is not None:

                    elapsed = (
                        time.time()
                        - hold_start_time
                    )

                else:

                    elapsed = 0


                # =================================================
                # ADD AFTER 1.5 SEC
                # =================================================

                if (
                    elapsed >= HOLD_DURATION
                    and prediction != last_added_word
                ):

                    sentence.append(
                        prediction
                    )

                    last_added_word = prediction

                    hold_start_time = time.time()


                    # =============================================
                    # EMERGENCY CHECK
                    # =============================================

                    prediction_key = str(
                        prediction
                    ).lower().strip()


                    if (
                        prediction_key
                        in EMERGENCY_SIGNS
                    ):

                        emergency_mode = True

                        emergency_message = (
                            EMERGENCY_SIGNS[
                                prediction_key
                            ]
                        )


                        print()

                        print(
                            "=" * 60
                        )

                        print(
                            "🚨 EMERGENCY MODE ACTIVATED!"
                        )

                        print(
                            "=" * 60
                        )

                        print(
                            f"Emergency Message: "
                            f"{emergency_message}"
                        )

                        print(
                            "=" * 60
                        )


                        threading.Thread(
                            target=speak_text,
                            args=(
                                emergency_message,
                            )
                        ).start()


                    # =============================================
                    # AMBIGUOUS INFORMATION
                    # =============================================

                    if ambiguous_mode:

                        print()

                        print(
                            "AMBIGUOUS SIGN DETECTED"
                        )

                        print(
                            "Top Predictions:"
                        )

                        for sign, conf in (
                            top_predictions
                        ):

                            print(
                                f"  {sign}: "
                                f"{conf:.1f}%"
                            )

                        print(
                            f"Final Decision: "
                            f"{prediction}"
                        )

                        print()


            # =================================================
            # NEW SIGN
            # =================================================

            else:

                current_word = prediction

                hold_start_time = time.time()

                last_added_word = ""


        # ====================================================
        # NO SIGN
        # ====================================================

        else:

            current_word = ""

            hold_start_time = None

            last_added_word = ""


        # ====================================================
        # UI
        # ====================================================

        h, w, _ = frame.shape


        # ====================================================
        # TOP BAR
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (w, 105),
            (0, 0, 0),
            -1
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        if prediction:

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


            # Detecting
            cv2.putText(
                frame,
                f"Detecting: "
                f"{prediction} "
                f"({confidence:.0f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )


            # =================================================
            # AMBIGUOUS UI
            # =================================================

            if ambiguous_mode:

                cv2.putText(
                    frame,
                    "AMBIGUOUS - CONTEXT CHECK",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )


                if len(top_predictions) >= 2:

                    p1, c1 = (
                        top_predictions[0]
                    )

                    p2, c2 = (
                        top_predictions[1]
                    )


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


            # =================================================
            # HOLD PROGRESS BAR
            # =================================================

            cv2.rectangle(
                frame,
                (10, 88),
                (310, 101),
                (100, 100, 100),
                1
            )


            bar_width = int(
                300 * hold_progress
            )


            cv2.rectangle(
                frame,
                (10, 88),
                (
                    10 + bar_width,
                    101
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
        # FACIAL MARKER
        # ====================================================

        if facial_marker:

            cv2.putText(
                frame,
                f"Face Marker: "
                f"{facial_marker}",
                (w - 300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )


        # ====================================================
        # EMERGENCY UI
        # ====================================================

        if emergency_mode:

            cv2.rectangle(
                frame,
                (0, 110),
                (w, 195),
                (0, 0, 255),
                -1
            )


            cv2.putText(
                frame,
                "!!! EMERGENCY MODE !!!",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                3
            )


            cv2.putText(
                frame,
                emergency_message,
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


        # ====================================================
        # SENTENCE BAR
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
        # CAMERA WINDOW
        # ====================================================

        cv2.imshow(
            "Sign Language Recognition App",
            frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # ====================================================
        # QUIT
        # ====================================================

        if key == ord("q"):

            break


        # ====================================================
        # CLEAR
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
                    f"Speaking: "
                    f"{full_text}"
                )

                threading.Thread(
                    target=speak_text,
                    args=(full_text,)
                ).start()


        # ====================================================
        # RESET EMERGENCY
        # ====================================================

        elif key == ord("e"):

            emergency_mode = False

            emergency_message = ""

            print(
                "Emergency Mode reset."
            )


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