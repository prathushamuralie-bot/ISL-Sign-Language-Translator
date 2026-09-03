"""
Sign Language Recognition - Data Collection Tool (TWO HANDS support)
=========================================================================
Idhu tool rendu kai um detect pannum. Ovvoru kai ku 21 landmarks (x,y,z) = 63 values,
rendu kai ku total 126 values. Oru kai mattum kaatina, andha kai matching slot
(Left/Right) la values irukkum, matha kai slot zeros ah irukkum.

CONTROLS:
- 'n' -> Next sign
- 'p' -> Previous sign
- 's' -> Capture sample
- 'q' -> Quit (save aagidum)
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import csv
import os

# ============ CONFIGURATION ============
SIGN_WORDS = [
    "Hello", "Good Morning", "Good Afternoon", "Good Night",
    "Thank You", "Yes", "No", "Please", "Sorry", "Help",
    "I Love You", "Bye","I","indian","sign","language","you","deaf","Man","Woman","Hearing","my","name"
]
SAMPLES_TARGET = 200
CSV_FILE = "dataset/sign_data_two_hands.csv"   # PUTHU FILE - old dataset vera, mix aagama irukka
MODEL_PATH = "hand_landmarker.task"
# ========================================

os.makedirs("dataset", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: '{MODEL_PATH}' file kedaikala!")
    exit(1)

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,                     # RENDU KAI - idhu than main change
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    running_mode=vision.RunningMode.IMAGE
)
landmarker = vision.HandLandmarker.create_from_options(options)

file_exists = os.path.isfile(CSV_FILE)


def normalize_single_hand(landmarks):
    """Oru kai oda 21 landmarks ah, wrist (point 0) ku relative ah normalize pannurom."""
    base_x, base_y = landmarks[0].x, landmarks[0].y
    normalized = []
    for lm in landmarks:
        normalized.extend([lm.x - base_x, lm.y - base_y, lm.z])
    return normalized


def extract_two_hand_features(result):
    """
    Result la irukura hands ah, Left/Right handedness paathu,
    fixed 126-length feature vector ah construct pannurom.
    Left hand -> first 63 values, Right hand -> next 63 values.
    Kai illana andha part zeros ah irukkum.
    """
    left_features = [0.0] * 63
    right_features = [0.0] * 63
    detected_any = False

    if result.hand_landmarks and result.handedness:
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            hand_label = handedness[0].category_name  # "Left" or "Right"
            feats = normalize_single_hand(landmarks)
            if hand_label == "Left":
                left_features = feats
            else:
                right_features = feats
            detected_any = True

    return left_features + right_features, detected_any


def draw_all_hands(frame, hand_landmarks_list):
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


def get_sample_counts():
    counts = {word: 0 for word in SIGN_WORDS}
    if os.path.isfile(CSV_FILE):
        with open(CSV_FILE, mode='r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and row[-1] in counts:
                    counts[row[-1]] += 1
    return counts


def draw_sidebar(frame, current_idx, counts):
    h, w, _ = frame.shape
    sidebar_width = 280
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - sidebar_width, 0), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "SIGN LIST", (w - sidebar_width + 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y = 65
    for i, word in enumerate(SIGN_WORDS):
        count = counts.get(word, 0)
        is_current = (i == current_idx)
        color = (0, 255, 255) if is_current else (200, 200, 200)
        prefix = "-> " if is_current else "   "
        text = f"{prefix}{word} ({count})"
        if is_current:
            cv2.rectangle(frame, (w - sidebar_width + 5, y - 20), (w - 5, y + 8), (70, 70, 0), -1)
        cv2.putText(frame, text, (w - sidebar_width + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y += 32

    cv2.putText(frame, "n=next  p=prev", (w - sidebar_width + 15, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "s=capture  q=quit", (w - sidebar_width + 15, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def main():
    cap = cv2.VideoCapture(0)
    current_idx = 0
    counts = get_sample_counts()

    csv_file = open(CSV_FILE, mode='a', newline='')
    csv_writer = csv.writer(csv_file)

    if not file_exists:
        header = []
        for side in ["L", "R"]:
            for i in range(21):
                header.extend([f'{side}_x{i}', f'{side}_y{i}', f'{side}_z{i}'])
        header.append('label')
        csv_writer.writerow(header)

    print("Two-hand data collection tool ready!")
    print("Controls: n=next sign, p=previous sign, s=capture sample, q=quit")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Camera access panna mudiyala.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        features, detected = None, False
        if result.hand_landmarks:
            draw_all_hands(frame, result.hand_landmarks)
            features, detected = extract_two_hand_features(result)

        current_word = SIGN_WORDS[current_idx]
        num_hands_detected = len(result.hand_landmarks) if result.hand_landmarks else 0
        cv2.putText(frame, f"Recording: {current_word}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {counts[current_word]}/{SAMPLES_TARGET} | Hands detected: {num_hands_detected}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        draw_sidebar(frame, current_idx, counts)
        cv2.imshow('Two-Hand Sign Language Data Collection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and detected:
            row = features + [current_word]
            csv_writer.writerow(row)
            csv_file.flush()
            counts[current_word] += 1
            print(f"Sample {counts[current_word]} captured for '{current_word}' ({num_hands_detected} hand(s))")
        elif key == ord('n'):
            current_idx = (current_idx + 1) % len(SIGN_WORDS)
        elif key == ord('p'):
            current_idx = (current_idx - 1) % len(SIGN_WORDS)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print("\nSession ended. Saved in", CSV_FILE)
    print("Final counts:", counts)


if __name__ == "__main__":
    main()
