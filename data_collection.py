"""
Sign Language Recognition - Smart Data Collection Tool
=========================================================
Idhu tool la ellaa sign words um oru list ah screen sidebar la kaanum.
Code edit pannama, key press panni easy ah oru sign la irundhu vera sign ku switch pannalam.

CONTROLS:
- 'n' key -> Next sign ku pogum
- 'p' key -> Previous sign ku pogum
- 's' key -> Current sign ku sample capture pannum
- 'q' key -> Exit pannum (ellame save aagidum)

Ovvoru sign ku already ethana samples collect aayirukku nu, sidebar la count kaanum.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import csv
import os

# ============ CONFIGURATION - Idha unga words ku maathikkalam ============
SIGN_WORDS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z"
]
SAMPLES_TARGET = 200          # ovvoru sign ku evlo samples venum
CSV_FILE = "dataset/sign_data.csv"
MODEL_PATH = "hand_landmarker.task"
# ===========================================================================

os.makedirs("dataset", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: '{MODEL_PATH}' file kedaikala!")
    print("Idha download pannunga: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
    exit(1)

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    running_mode=vision.RunningMode.IMAGE
)
landmarker = vision.HandLandmarker.create_from_options(options)

file_exists = os.path.isfile(CSV_FILE)


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


def get_sample_counts():
    """CSV file padichi, ovvoru sign ku ethana samples already irukku nu count pannurom."""
    counts = {word: 0 for word in SIGN_WORDS}
    if os.path.isfile(CSV_FILE):
        with open(CSV_FILE, mode='r') as f:
            reader = csv.reader(f)
            next(reader, None)  # header skip pannunga
            for row in reader:
                if row and row[-1] in counts:
                    counts[row[-1]] += 1
    return counts


def draw_sidebar(frame, current_idx, counts):
    """Screen right side la ellaa sign words um list pannurom, current one highlight pannurom."""
    h, w, _ = frame.shape
    sidebar_width = 280
    # Semi-transparent sidebar background
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
            cv2.rectangle(frame, (w - sidebar_width + 5, y - 20),
                           (w - 5, y + 8), (70, 70, 0), -1)

        cv2.putText(frame, text, (w - sidebar_width + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y += 32

    # Controls reminder kீழ la
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
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        header.append('label')
        csv_writer.writerow(header)

    print("Data collection tool ready!")
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

        landmark_list = []
        if result.hand_landmarks:
            draw_hand_landmarks(frame, result.hand_landmarks)
            landmark_list = [(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]]

        current_word = SIGN_WORDS[current_idx]
        cv2.putText(frame, f"Recording: {current_word}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {counts[current_word]}/{SAMPLES_TARGET}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        draw_sidebar(frame, current_idx, counts)

        cv2.imshow('Sign Language Data Collection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and landmark_list:
            normalized = normalize_landmarks(landmark_list)
            normalized.append(current_word)
            csv_writer.writerow(normalized)
            csv_file.flush()
            counts[current_word] += 1
            print(f"Sample {counts[current_word]} captured for '{current_word}'")
        elif key == ord('n'):
            current_idx = (current_idx + 1) % len(SIGN_WORDS)
            print(f"Switched to: {SIGN_WORDS[current_idx]}")
        elif key == ord('p'):
            current_idx = (current_idx - 1) % len(SIGN_WORDS)
            print(f"Switched to: {SIGN_WORDS[current_idx]}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print("\nSession ended. All samples saved in", CSV_FILE)
    print("Final counts:", counts)


if __name__ == "__main__":
    main()