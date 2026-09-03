"""
Sign Language Recognition - ALPHABET Data Collection Tool (A-Z, Auto-Capture)
====================================================================================
Idhu tool, A to Z alphabets ku data collect pannum - "Spelling Mode" feature ku
(e.g., P-R-I-Y-A spell panni "PRIYA" nu word build pannuradhu).

CONTROLS:
- 'n' -> Next letter
- 'p' -> Previous letter
- 'a' -> Auto-capture mode ON/OFF toggle
- 's' -> Manual capture
- 'q' -> Quit
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import csv
import os
import time
import string

# ============ CONFIGURATION ============
SIGN_WORDS = list(string.ascii_uppercase)   # A, B, C, ... Z
SAMPLES_TARGET = 150
CSV_FILE = "dataset/alphabet_data.csv"
MODEL_PATH = "hand_landmarker.task"
CAPTURE_INTERVAL = 1.5
# ========================================

os.makedirs("dataset", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: '{MODEL_PATH}' file kedaikala!")
    exit(1)

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    running_mode=vision.RunningMode.IMAGE
)
landmarker = vision.HandLandmarker.create_from_options(options)

file_exists = os.path.isfile(CSV_FILE)


def normalize_single_hand(landmarks):
    base_x, base_y = landmarks[0].x, landmarks[0].y
    normalized = []
    for lm in landmarks:
        normalized.extend([lm.x - base_x, lm.y - base_y, lm.z])
    return normalized


def extract_two_hand_features(result):
    left_features = [0.0] * 63
    right_features = [0.0] * 63
    detected_any = False
    if result.hand_landmarks and result.handedness:
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            hand_label = handedness[0].category_name
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


def draw_sidebar(frame, current_idx, counts, auto_mode):
    h, w, _ = frame.shape
    sidebar_width = 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - sidebar_width, 0), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "A-Z LETTERS", (w - sidebar_width + 15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Grid layout - 3 columns of letters (compact ah kaatta)
    cols = 3
    start_y = 55
    col_width = sidebar_width // cols
    for i, letter in enumerate(SIGN_WORDS):
        col = i % cols
        row = i // cols
        x = w - sidebar_width + 10 + col * col_width
        y = start_y + row * 22
        is_current = (i == current_idx)
        count = counts.get(letter, 0)
        color = (0, 255, 255) if is_current else ((0, 200, 0) if count >= SAMPLES_TARGET else (180, 180, 180))
        text = f"{letter}:{count}"
        if is_current:
            cv2.rectangle(frame, (x - 3, y - 15), (x + col_width - 8, y + 5), (70, 70, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    mode_text = "AUTO: ON" if auto_mode else "AUTO: OFF"
    mode_color = (0, 255, 0) if auto_mode else (150, 150, 150)
    cv2.putText(frame, mode_text, (w - sidebar_width + 15, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)
    cv2.putText(frame, "a=auto n/p=switch", (w - sidebar_width + 15, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
    cv2.putText(frame, "s=manual q=quit", (w - sidebar_width + 15, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)


def main():
    cap = cv2.VideoCapture(0)
    current_idx = 0
    counts = get_sample_counts()
    auto_mode = False
    auto_timer_start = None
    flash_until = 0

    csv_file = open(CSV_FILE, mode='a', newline='')
    csv_writer = csv.writer(csv_file)

    if not file_exists:
        header = []
        for side in ["L", "R"]:
            for i in range(21):
                header.extend([f'{side}_x{i}', f'{side}_y{i}', f'{side}_z{i}'])
        header.append('label')
        csv_writer.writerow(header)

    print("Alphabet data collection tool ready!")
    print("Controls: n=next, p=previous, a=toggle auto-capture, s=manual capture, q=quit")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        features, detected = None, False
        if result.hand_landmarks:
            draw_all_hands(frame, result.hand_landmarks)
            features, detected = extract_two_hand_features(result)

        current_letter = SIGN_WORDS[current_idx]
        num_hands_detected = len(result.hand_landmarks) if result.hand_landmarks else 0

        cv2.putText(frame, f"Letter: {current_letter}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {counts[current_letter]}/{SAMPLES_TARGET} | Hands: {num_hands_detected}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if auto_mode and detected:
            if auto_timer_start is None:
                auto_timer_start = time.time()
            elapsed = time.time() - auto_timer_start
            remaining = CAPTURE_INTERVAL - elapsed
            if remaining > 0:
                countdown_num = int(remaining) + 1
                cv2.putText(frame, str(countdown_num), (frame.shape[1] // 2 - 200, frame.shape[1] // 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 6)
            else:
                row = features + [current_letter]
                csv_writer.writerow(row)
                csv_file.flush()
                counts[current_letter] += 1
                print(f"[AUTO] Sample {counts[current_letter]} captured for '{current_letter}'")
                auto_timer_start = time.time()
                flash_until = time.time() + 0.3
        elif auto_mode and not detected:
            auto_timer_start = None
            cv2.putText(frame, "Show your hand(s)...", (frame.shape[1] // 2 - 180, frame.shape[1] // 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        else:
            auto_timer_start = None

        if time.time() < flash_until:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.putText(frame, "CAPTURED!", (frame.shape[1] // 2 - 150, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        draw_sidebar(frame, current_idx, counts, auto_mode)
        cv2.imshow('Alphabet Data Collection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and detected and not auto_mode:
            row = features + [current_letter]
            csv_writer.writerow(row)
            csv_file.flush()
            counts[current_letter] += 1
            print(f"[MANUAL] Sample {counts[current_letter]} captured for '{current_letter}'")
            flash_until = time.time() + 0.3
        elif key == ord('n'):
            current_idx = (current_idx + 1) % len(SIGN_WORDS)
            auto_timer_start = None
        elif key == ord('p'):
            current_idx = (current_idx - 1) % len(SIGN_WORDS)
            auto_timer_start = None
        elif key == ord('a'):
            auto_mode = not auto_mode
            auto_timer_start = None
            print(f"Auto mode: {'ON' if auto_mode else 'OFF'}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print("\nSession ended. Saved in", CSV_FILE)
    print("Final counts:", counts)


if __name__ == "__main__":
    main()
