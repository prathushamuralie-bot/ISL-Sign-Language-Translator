"""
Sign Language Recognition - Data Collection Tool (AUTO-CAPTURE MODE, Two Hands)
====================================================================================
Idhu tool la "Auto Capture" mode irukku - rendu kai um camera munnadi vachi irundhalum,
key press pannama, automatic ah samples capture aagum (countdown vachi).

CONTROLS:
- 'n' -> Next sign
- 'p' -> Previous sign
- 'a' -> Auto-capture mode ON/OFF toggle pannum
- 's' -> Manual capture (auto mode off na use pannalam)
- 'q' -> Quit (save aagidum)

AUTO MODE epadi work aagum:
- 'a' press pannina apparam, kai ah steady ah camera mun vachukonga
- Screen la countdown kaanum (3, 2, 1...)
- 0 aana automatic ah sample capture aagum, "CAPTURED!" flash aagum
- Idhu repeat aagikondu irukum (every CAPTURE_INTERVAL seconds), target count varaikkum
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import csv
import os
import time

# ============ CONFIGURATION ============
SIGN_WORDS = ["Hello", "Good Morning", "Good Afternoon", "Good Night",
    "Thank You", "Yes", "No", "Please", "Sorry", "Help",
    "I Love You", "Bye","I","indian","sign","language","you","deaf","Man","Woman","Hearing","my","name",
              "white","black","brown","green","red","blue","yellow","orange","gold","pink","is",
              "week","day","make","app","talk","team","malayalam","hindi","tamil","english","kannada","telugu"
]
SAMPLES_TARGET = 200
CSV_FILE = "dataset/sign_data_two_hands.csv"
MODEL_PATH = "hand_landmarker.task"
CAPTURE_INTERVAL = 2.0    # evlo seconds ku oru sample auto capture aaganum
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

    mode_text = "AUTO MODE: ON" if auto_mode else "AUTO MODE: OFF"
    mode_color = (0, 255, 0) if auto_mode else (150, 150, 150)
    cv2.putText(frame, mode_text, (w - sidebar_width + 15, h - 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)
    cv2.putText(frame, "a=toggle auto  n/p=switch", (w - sidebar_width + 15, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(frame, "s=manual  q=quit", (w - sidebar_width + 15, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)


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

    print("Auto-capture data collection tool ready!")
    print("Controls: n=next, p=previous, a=toggle auto-capture, s=manual capture, q=quit")

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
        cv2.putText(frame, f"Samples: {counts[current_word]}/{SAMPLES_TARGET} | Hands: {num_hands_detected}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ===== AUTO CAPTURE LOGIC =====
        if auto_mode and detected:
            if auto_timer_start is None:
                auto_timer_start = time.time()

            elapsed = time.time() - auto_timer_start
            remaining = CAPTURE_INTERVAL - elapsed

            if remaining > 0:
                # Countdown number periya ah screen naduvula kaatunga
                countdown_num = int(remaining) + 1
                cv2.putText(frame, str(countdown_num), (frame.shape[1] // 2 - 250, frame.shape[1] // 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 6)
            else:
                # Capture pannurom
                row = features + [current_word]
                csv_writer.writerow(row)
                csv_file.flush()
                counts[current_word] += 1
                print(f"[AUTO] Sample {counts[current_word]} captured for '{current_word}'")
                auto_timer_start = time.time()
                flash_until = time.time() + 0.3

        elif auto_mode and not detected:
            # Kai kaanaama irundha, timer reset pannurom (hand vandhu apparam fresh ah start aagum)
            auto_timer_start = None
            cv2.putText(frame, "Show your hand(s)...", (frame.shape[1] // 2 - 180, frame.shape[1] // 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        else:
            auto_timer_start = None

        # Flash effect - capture aana udane green flash
        if time.time() < flash_until:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.putText(frame, "CAPTURED!", (frame.shape[1] // 2 - 150, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        draw_sidebar(frame, current_idx, counts, auto_mode)
        cv2.imshow('Auto-Capture Sign Language Data Collection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and detected and not auto_mode:
            row = features + [current_word]
            csv_writer.writerow(row)
            csv_file.flush()
            counts[current_word] += 1
            print(f"[MANUAL] Sample {counts[current_word]} captured for '{current_word}'")
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
