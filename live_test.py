"""
Sign Language Recognition - Live Testing Script
===================================================
Idhu script webcam open pannitu, real-time la unga kai sign ah detect pannitu,
train pannina model use panni, "idhu enna sign" nu predict pannum, screen la display pannum.

Idha run panna munnadi "train_model.py" already run pannirukanum.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pickle
import numpy as np
import os

MODEL_PATH = "model/sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: '{MODEL_PATH}' kedaikala. Munnadi 'train_model.py' run pannunga.")
    exit(1)

if not os.path.exists(HAND_MODEL_PATH):
    print(f"ERROR: '{HAND_MODEL_PATH}' kedaikala.")
    exit(1)

# Trained model load pannurom
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# MediaPipe hand landmarker setup
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


def main():
    cap = cv2.VideoCapture(0)

    print("Live testing start aaguthu. Kai kaatunga, prediction screen la varum.")
    print("'q' press panna exit aagum.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Camera access panna mudiyala.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        prediction_text = "No hand detected"
        confidence_text = ""

        if result.hand_landmarks:
            draw_hand_landmarks(frame, result.hand_landmarks)
            landmark_list = [(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]]
            normalized = normalize_landmarks(landmark_list)

            # Model ku input format ah convert pannurom
            input_data = np.array(normalized).reshape(1, -1)

            # Prediction pannurom
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            confidence = max(probabilities) * 100

            prediction_text = f"Sign: {prediction}"
            confidence_text = f"Confidence: {confidence:.1f}%"

        # Screen la results display pannurom
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        cv2.putText(frame, prediction_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, confidence_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Sign Language Live Test', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
