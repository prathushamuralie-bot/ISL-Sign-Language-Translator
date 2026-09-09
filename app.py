"""
============================================================
        ISL SIGN LANGUAGE TRANSLATOR
============================================================

FEATURES
--------
1. Two-Hand Sign Recognition
2. 126 Feature Random Forest Model
3. Smart Context / Sentence Building
4. Ambiguous Sign Disambiguation
5. Emergency Communication
6. Text-to-Speech
7. Sentence Clear
8. Backspace
9. Voice Output

CONTROLS
--------
c = Clear sentence
v = Voice
b = Backspace
e = Reset Emergency Mode
q = Quit

MODEL
-----
sign_model.pkl
126 features:
21 landmarks x 3 coordinates x 2 hands
============================================================
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import numpy as np
import pickle
import os
import time
import threading

from gtts import gTTS


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model/sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"

CONFIDENCE_THRESHOLD = 70.0
HOLD_DURATION = 1.5

# Difference between first and second prediction
# <= this value -> possible ambiguous sign
AMBIGUITY_THRESHOLD = 15.0


# ============================================================
# EMERGENCY SIGNS
# ============================================================

EMERGENCY_SIGNS = {

    "help":
        "I need help.",

    "doctor":
        "I need a doctor.",

    "emergency":
        "Emergency.",

    "family":
        "Please call my family.",

    "ambulance":
        "Please call an ambulance.",

    "hospital":
        "I need to go to the hospital.",

    "pain":
        "I am in pain."
}


# ============================================================
# SMART CONTEXT PHRASES
# ============================================================

CONTEXT_PHRASES = {

    # --------------------------------------------------------
    # BASIC NEEDS
    # --------------------------------------------------------

    ("i", "need", "water"):
        "I need water.",

    ("i", "want", "water"):
        "I want water.",

    ("i", "need", "food"):
        "I need food.",

    ("i", "want", "food"):
        "I want food.",

    ("i", "need", "help"):
        "I need help.",

    ("i", "need", "doctor"):
        "I need a doctor.",

    ("i", "need", "medicine"):
        "I need medicine.",

    ("i", "need", "hospital"):
        "I need a hospital.",


    # --------------------------------------------------------
    # EMERGENCY
    # --------------------------------------------------------

    ("call", "ambulance"):
        "Please call an ambulance.",

    ("call", "doctor"):
        "Please call a doctor.",

    ("call", "family"):
        "Please call my family.",

    ("help", "me"):
        "Please help me.",

    ("i", "am", "pain"):
        "I am in pain.",


    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    ("where", "hospital"):
        "Where is the hospital?",

    ("where", "bathroom"):
        "Where is the bathroom?",

    ("where", "school"):
        "Where is the school?",

    ("where", "bus"):
        "Where is the bus?",

    ("where", "home"):
        "Where is home?",


    # --------------------------------------------------------
    # TRAVEL
    # --------------------------------------------------------

    ("i", "want", "go"):
        "I want to go.",

    ("i", "want", "home"):
        "I want to go home.",

    ("i", "want", "school"):
        "I want to go to school.",

    ("i", "want", "hospital"):
        "I want to go to the hospital.",


    # --------------------------------------------------------
    # SOCIAL
    # --------------------------------------------------------

    ("what", "your", "name"):
        "What is your name.",

    ("where", "you"):
        "Where are you?",

    ("how", "are", "you"):
        "How are you?",

    ("thank", "you"):
        "Thank you.",

    ("good", "morning"):
        "Good morning.",

    ("good", "night"):
        "Good night.",


    # --------------------------------------------------------
    # QUESTIONS
    # --------------------------------------------------------

    ("what", "this"):
        "What is this?",

    ("what", "that"):
        "What is that?",

    ("where", "this"):
        "Where is this?",

    ("why", "you"):
        "Why are you?"
}


# ============================================================
# AMBIGUOUS SIGN GROUPS
# ============================================================

AMBIGUOUS_SIGN_GROUPS = [

    {"can", "can't"},

    {"yes", "no"},

    {"good", "bad"},

    {"come", "go"},

    {"here", "there"},

    {"this", "that"}

]


# ============================================================
# FILE CHECK
# ============================================================

if not os.path.exists(MODEL_PATH):

    print()
    print("ERROR: Model file not found!")
    print()
    print(
        f"Expected: {MODEL_PATH}"
    )
    print()
    print(
        "First run train_model.py"
    )
    print()

    exit()


if not os.path.exists(HAND_MODEL_PATH):

    print()
    print("ERROR: hand_landmarker.task not found!")
    print()
    print(
        "Place hand_landmarker.task in the project folder."
    )
    print()

    exit()


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

print()
print("Loading sign recognition model...")

with open(
    MODEL_PATH,
    "rb"
) as f:

    model = pickle.load(f)


print(
    "Model loaded successfully!"
)


# ============================================================
# CHECK MODEL FEATURE COUNT
# ============================================================

EXPECTED_FEATURES = model.n_features_in_

print(
    f"Model expects {EXPECTED_FEATURES} features."
)


if EXPECTED_FEATURES != 126:

    print()
    print(
        "WARNING:"
    )

    print(
        "This app is designed for a 126-feature "
        "two-hand model."
    )

    print(
        f"Current model expects: "
        f"{EXPECTED_FEATURES}"
    )

    print()


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

base_options = mp_python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)


options = vision.HandLandmarkerOptions(

    base_options=base_options,

    # IMPORTANT
    # Two hands
    num_hands=2,

    min_hand_detection_confidence=0.4,

    min_hand_presence_confidence=0.4,

    min_tracking_confidence=0.4,

    running_mode=vision.RunningMode.IMAGE
)


landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# HAND LANDMARK -> 63 FEATURES
# ============================================================

def hand_to_features(hand_landmarks):

    """
    One hand:

    21 landmarks
    x,y,z

    = 63 features
    """

    features = []

    # No hand
    if hand_landmarks is None:

        return [0.0] * 63


    # Wrist is landmark 0
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    base_z = hand_landmarks[0].z


    for landmark in hand_landmarks:

        features.extend([

            landmark.x - base_x,

            landmark.y - base_y,

            landmark.z - base_z

        ])


    return features


# ============================================================
# GET LEFT + RIGHT HAND
# ============================================================

def get_left_right_hands(result):

    left_hand = None
    right_hand = None


    if not result.hand_landmarks:

        return left_hand, right_hand


    for index, hand_landmarks in enumerate(
        result.hand_landmarks
    ):

        if (
            result.handedness
            and index < len(result.handedness)
        ):

            handedness = (
                result
                .handedness[index][0]
                .category_name
                .lower()
            )


            if handedness == "left":

                left_hand = hand_landmarks


            elif handedness == "right":

                right_hand = hand_landmarks


    return left_hand, right_hand


# ============================================================
# CREATE TWO-HAND 126 FEATURES
# ============================================================

def create_two_hand_features(
    left_hand,
    right_hand
):

    """
    Left hand  = 63
    Right hand = 63

    Total = 126
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


    if len(features) != 126:

        raise ValueError(
            f"Expected 126 features, "
            f"but got {len(features)}"
        )


    return features


# ============================================================
# DRAW HAND LANDMARKS
# ============================================================

def draw_hand_landmarks(
    frame,
    hand_landmarks_list
):

    height, width, _ = frame.shape


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

        points = []


        for landmark in hand_landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            points.append(
                (x, y)
            )


        # Draw connections

        for start, end in connections:

            cv2.line(

                frame,

                points[start],

                points[end],

                (0, 255, 0),

                2
            )


        # Draw landmarks

        for point in points:

            cv2.circle(

                frame,

                point,

                4,

                (0, 0, 255),

                -1
            )


# ============================================================
# GET TOP PREDICTIONS
# ============================================================

def get_top_predictions(
    input_data,
    top_n=3
):

    probabilities = (
        model.predict_proba(
            input_data
        )[0]
    )


    classes = model.classes_


    top_indices = np.argsort(
        probabilities
    )[-top_n:][::-1]


    predictions = []


    for index in top_indices:

        label = str(
            classes[index]
        )


        confidence = (
            float(
                probabilities[index]
            )
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
# CHECK AMBIGUOUS
# ============================================================

def is_ambiguous(
    predictions
):

    if len(predictions) < 2:

        return False


    sign1, confidence1 = (
        predictions[0]
    )

    sign2, confidence2 = (
        predictions[1]
    )


    difference = abs(
        confidence1 - confidence2
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
# SMART CONTEXT SENTENCE BUILDER
# ============================================================

def build_smart_sentence(words):

    """
    Converts recognized signs
    into meaningful English sentences.
    """

    if not words:

        return ""


    normalized = [

        str(word)
        .lower()
        .strip()

        for word in words

    ]


    # --------------------------------------------------------
    # Remove consecutive duplicates
    # --------------------------------------------------------

    cleaned = []


    for word in normalized:

        if (
            not cleaned
            or word != cleaned[-1]
        ):

            cleaned.append(word)


    # --------------------------------------------------------
    # Exact phrase matching
    # --------------------------------------------------------

    for phrase, sentence in (
        CONTEXT_PHRASES.items()
    ):

        phrase_length = len(
            phrase
        )


        if len(cleaned) < phrase_length:

            continue


        for i in range(
            len(cleaned) - phrase_length + 1
        ):

            section = tuple(
                cleaned[
                    i:i + phrase_length
                ]
            )


            if section == phrase:

                return sentence


    # --------------------------------------------------------
    # I NEED
    # --------------------------------------------------------

    if (
        len(cleaned) >= 3
        and cleaned[0] == "i"
        and cleaned[1] == "need"
    ):

        return (
            "I need "
            + " ".join(
                cleaned[2:]
            )
            + "."
        )


    # --------------------------------------------------------
    # I WANT
    # --------------------------------------------------------

    if (
        len(cleaned) >= 3
        and cleaned[0] == "i"
        and cleaned[1] == "want"
    ):

        return (
            "I want "
            + " ".join(
                cleaned[2:]
            )
            + "."
        )


    # --------------------------------------------------------
    # WHERE
    # --------------------------------------------------------

    if (
        len(cleaned) >= 2
        and cleaned[0] == "where"
    ):

        return (
            "Where is "
            + " ".join(
                cleaned[1:]
            )
            + "?"
        )


    # --------------------------------------------------------
    # WHAT
    # --------------------------------------------------------

    if (
        len(cleaned) >= 2
        and cleaned[0] == "what"
    ):

        return (
            "What is "
            + " ".join(
                cleaned[1:]
            )
            + "?"
        )


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    sentence = " ".join(
        cleaned
    )


    if not sentence:

        return ""


    return (
        sentence[0].upper()
        + sentence[1:]
        + "."
    )


# ============================================================
# DISAMBIGUATE SIGN
# ============================================================

def disambiguate_sign(
    predictions,
    sentence_words
):

    if not predictions:

        return None


    best_sign = predictions[0][0]


    if len(predictions) < 2:

        return best_sign


    sign1, conf1 = (
        predictions[0]
    )

    sign2, conf2 = (
        predictions[1]
    )


    s1 = sign1.lower().strip()
    s2 = sign2.lower().strip()


    context = " ".join(
        sentence_words
    ).lower()


    # --------------------------------------------------------
    # CAN / CAN'T
    # --------------------------------------------------------

    if {s1, s2} == {"can", "can't"}:

        if (
            "not" in context
            or "never" in context
        ):

            if s1 == "can't":

                return sign1

            return sign2


    # --------------------------------------------------------
    # YES / NO
    # --------------------------------------------------------

    if {s1, s2} == {"yes", "no"}:

        if (
            "not" in context
            or "don't" in context
            or "dont" in context
        ):

            if s1 == "no":

                return sign1

            return sign2


    # --------------------------------------------------------
    # GOOD / BAD
    # --------------------------------------------------------

    if {s1, s2} == {"good", "bad"}:

        if (
            "not" in context
            or "bad" in context
        ):

            if s1 == "bad":

                return sign1

            return sign2


    # --------------------------------------------------------
    # COME / GO
    # --------------------------------------------------------

    if {s1, s2} == {"come", "go"}:

        if "here" in context:

            if s1 == "come":

                return sign1

            return sign2


        if "there" in context:

            if s1 == "go":

                return sign1

            return sign2


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return best_sign


# ============================================================
# TEXT TO SPEECH
# ============================================================

def speak_text(text):

    if not text:

        return


    try:

        print()
        print(
            f"🔊 Speaking: {text}"
        )


        tts = gTTS(

            text=text,

            lang="en"

        )


        filename = (
            "temp_speech.mp3"
        )


        tts.save(
            filename
        )


        os.system(
            f'start "" "{filename}"'
        )


    except Exception as error:

        print(
            f"TTS Error: {error}"
        )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(0)


    if not cap.isOpened():

        print(
            "ERROR: Camera open panna mudiyala."
        )

        return


    # ========================================================
    # SENTENCE
    # ========================================================

    sentence = []


    current_prediction = None

    hold_start_time = None

    last_added_prediction = None


    # ========================================================
    # EMERGENCY
    # ========================================================

    emergency_mode = False

    emergency_message = ""


    # ========================================================
    # DISPLAY VARIABLES
    # ========================================================

    top_predictions = []

    ambiguous_mode = False

    prediction = None

    confidence = 0.0


    # ========================================================
    # START MESSAGE
    # ========================================================

    print()
    print(
        "=" * 60
    )

    print(
        "          SIGN LANGUAGE RECOGNITION APP"
    )

    print(
        "=" * 60
    )

    print()

    print(
        "App ready!"
    )

    print(
        "Sign ah 1.5 seconds hold pannunga."
    )

    print(
        "Recognized sign sentence la add aagum."
    )

    print()

    print(
        "FEATURES:"
    )

    print(
        "✓ Two-Hand Recognition"
    )

    print(
        "✓ 126 Feature Model"
    )

    print(
        "✓ Smart Context"
    )

    print(
        "✓ Ambiguous Sign Disambiguation"
    )

    print(
        "✓ Emergency Communication"
    )

    print(
        "✓ Text-to-Speech"
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
        "e = Reset Emergency"
    )

    print(
        "q = Quit"
    )

    print()


    # ========================================================
    # CAMERA LOOP
    # ========================================================

    while cap.isOpened():

        success, frame = (
            cap.read()
        )


        if not success:

            print(
                "Camera frame read failed."
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

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb_frame

        )


        # ====================================================
        # MEDIAPIPE
        # ====================================================

        result = landmarker.detect(
            mp_image
        )


        prediction = None

        confidence = 0.0

        top_predictions = []

        ambiguous_mode = False


        # ====================================================
        # HANDS DETECTED
        # ====================================================

        if result.hand_landmarks:

            # Draw hands

            draw_hand_landmarks(

                frame,

                result.hand_landmarks

            )


            # Get left/right

            left_hand, right_hand = (
                get_left_right_hands(
                    result
                )
            )


            # =================================================
            # CREATE 126 FEATURES
            # =================================================

            features = (
                create_two_hand_features(

                    left_hand,

                    right_hand

                )
            )


            # =================================================
            # MODEL INPUT
            # =================================================

            input_data = np.array(
                features,
                dtype=np.float32
            ).reshape(
                1,
                -1
            )


            # Safety

            if input_data.shape[1] != (
                EXPECTED_FEATURES
            ):

                print(
                    "Feature mismatch!"
                )

                print(
                    f"Created: "
                    f"{input_data.shape[1]}"
                )

                print(
                    f"Expected: "
                    f"{EXPECTED_FEATURES}"
                )

                continue


            # =================================================
            # TOP PREDICTIONS
            # =================================================

            top_predictions = (
                get_top_predictions(
                    input_data,
                    top_n=3
                )
            )


            if top_predictions:

                best_sign = (
                    top_predictions[0][0]
                )

                best_confidence = (
                    top_predictions[0][1]
                )


                if (
                    best_confidence
                    >= CONFIDENCE_THRESHOLD
                ):

                    prediction = (
                        best_sign
                    )

                    confidence = (
                        best_confidence
                    )


                    # =========================================
                    # AMBIGUOUS
                    # =========================================

                    ambiguous_mode = (
                        is_ambiguous(
                            top_predictions
                        )
                    )


                    if ambiguous_mode:

                        prediction = (
                            disambiguate_sign(

                                top_predictions,

                                sentence

                            )
                        )


                        # Get confidence
                        for sign, conf in (
                            top_predictions
                        ):

                            if sign == prediction:

                                confidence = conf

                                break


        # ====================================================
        # HOLD LOGIC
        # ====================================================

        if prediction:

            # Same sign

            if (
                prediction
                == current_prediction
            ):

                if hold_start_time is None:

                    hold_start_time = (
                        time.time()
                    )


                elapsed = (
                    time.time()
                    - hold_start_time
                )


                # =============================================
                # ADD SIGN
                # =============================================

                if (
                    elapsed
                    >= HOLD_DURATION
                ):

                    if (
                        prediction
                        != last_added_prediction
                    ):

                        sentence.append(
                            prediction
                        )


                        last_added_prediction = (
                            prediction
                        )


                        # =====================================
                        # SMART SENTENCE
                        # =====================================

                        smart_sentence = (
                            build_smart_sentence(
                                sentence
                            )
                        )


                        print()

                        print(
                            "-" * 60
                        )

                        print(
                            f"Recognized Sign: "
                            f"{prediction}"
                        )

                        print(
                            f"Sentence: "
                            f"{smart_sentence}"
                        )

                        print(
                            "-" * 60
                        )


                        # =====================================
                        # EMERGENCY
                        # =====================================

                        prediction_key = (
                            str(
                                prediction
                            )
                            .lower()
                            .strip()
                        )


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
                                "🚨 "
                                "EMERGENCY MODE ACTIVATED!"
                            )

                            print(
                                emergency_message
                            )


                            threading.Thread(

                                target=speak_text,

                                args=(
                                    emergency_message,
                                )

                            ).start()


                        # =====================================
                        # AMBIGUOUS LOG
                        # =====================================

                        if ambiguous_mode:

                            print()

                            print(
                                "⚠ AMBIGUOUS SIGN"
                            )

                            for sign, conf in (
                                top_predictions
                            ):

                                print(
                                    f"{sign}: "
                                    f"{conf:.1f}%"
                                )

                            print(
                                f"Selected: "
                                f"{prediction}"
                            )


            # =================================================
            # NEW SIGN
            # =================================================

            else:

                current_prediction = (
                    prediction
                )

                hold_start_time = (
                    time.time()
                )

                last_added_prediction = None


        # ====================================================
        # NO SIGN
        # ====================================================

        else:

            current_prediction = None

            hold_start_time = None

            last_added_prediction = None


        # ====================================================
        # UI SIZE
        # ====================================================

        height, width, _ = (
            frame.shape
        )


        # ====================================================
        # TOP BAR
        # ====================================================

        cv2.rectangle(

            frame,

            (0, 0),

            (width, 115),

            (0, 0, 0),

            -1

        )


        # ====================================================
        # PREDICTION TEXT
        # ====================================================

        if prediction:

            cv2.putText(

                frame,

                f"Sign: {prediction}",

                (10, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.75,

                (0, 255, 0),

                2

            )


            cv2.putText(

                frame,

                f"Confidence: "
                f"{confidence:.1f}%",

                (10, 58),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 255),

                2

            )


            # =================================================
            # AMBIGUOUS UI
            # =================================================

            if ambiguous_mode:

                cv2.putText(

                    frame,

                    "AMBIGUOUS - "
                    "CONTEXT CHECK",

                    (10, 84),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (0, 255, 255),

                    2

                )


            # =================================================
            # HOLD PROGRESS
            # =================================================

            if hold_start_time:

                elapsed = (
                    time.time()
                    - hold_start_time
                )

                progress = min(

                    elapsed
                    / HOLD_DURATION,

                    1.0

                )

            else:

                progress = 0.0


            bar_width = int(
                300 * progress
            )


            cv2.rectangle(

                frame,

                (width - 320, 15),

                (width - 20, 32),

                (100, 100, 100),

                2

            )


            cv2.rectangle(

                frame,

                (width - 320, 15),

                (
                    width - 320 + bar_width,
                    32
                ),

                (0, 255, 255),

                -1

            )


        else:

            cv2.putText(

                frame,

                "Show a sign...",

                (10, 35),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (180, 180, 180),

                2

            )


        # ====================================================
        # EMERGENCY DISPLAY
        # ====================================================

        if emergency_mode:

            cv2.rectangle(

                frame,

                (0, 120),

                (width, 205),

                (0, 0, 180),

                -1

            )


            cv2.putText(

                frame,

                "!!! EMERGENCY MODE !!!",

                (20, 155),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.85,

                (255, 255, 255),

                3

            )


            cv2.putText(

                frame,

                emergency_message,

                (20, 190),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.62,

                (255, 255, 255),

                2

            )


        # ====================================================
        # SENTENCE
        # ====================================================

        smart_sentence = (
            build_smart_sentence(
                sentence
            )
        )


        # Bottom area

        cv2.rectangle(

            frame,

            (0, height - 100),

            (width, height),

            (30, 30, 30),

            -1

        )


        cv2.putText(

            frame,

            "Sentence:",

            (10, height - 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 255, 255),

            2

        )


        # Avoid too long text

        display_sentence = (
            smart_sentence
        )


        if len(display_sentence) > 70:

            display_sentence = (
                display_sentence[-70:]
            )


        cv2.putText(

            frame,

            display_sentence,

            (10, height - 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        # ====================================================
        # CONTROL INFO
        # ====================================================

        cv2.putText(

            frame,

            "C:Clear  V:Voice  "
            "B:Backspace  E:Emergency  Q:Quit",

            (10, height - 8),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.4,

            (180, 180, 180),

            1

        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(

            "ISL Sign Language Translator",

            frame

        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = (
            cv2.waitKey(1)
            & 0xFF
        )


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

            current_prediction = None

            hold_start_time = None

            last_added_prediction = None


            print()

            print(
                "✓ Sentence cleared."
            )


        # ====================================================
        # VOICE
        # ====================================================

        elif key == ord("v"):

            if sentence:

                smart_sentence = (
                    build_smart_sentence(
                        sentence
                    )
                )


                threading.Thread(

                    target=speak_text,

                    args=(
                        smart_sentence,
                    )

                ).start()


            else:

                print(
                    "Sentence empty."
                )


        # ====================================================
        # BACKSPACE
        # ====================================================

        elif key == ord("b"):

            if sentence:

                removed = (
                    sentence.pop()
                )


                print(
                    f"Removed: {removed}"
                )


                if sentence:

                    print(
                        "Sentence: "
                        + build_smart_sentence(
                            sentence
                        )
                    )

                else:

                    print(
                        "Sentence empty."
                    )


        # ====================================================
        # RESET EMERGENCY
        # ====================================================

        elif key == ord("e"):

            emergency_mode = False

            emergency_message = ""


            print()

            print(
                "✓ Emergency Mode reset."
            )


    # ========================================================
    # RELEASE CAMERA
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()