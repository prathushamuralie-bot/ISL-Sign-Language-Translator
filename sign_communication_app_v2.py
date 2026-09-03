"""
Sign Language Communication App - Modern UI/UX Version
============================================================
Idhu app oda design version - customtkinter use panni, modern, polished UI.
Same detection logic, aana professional look and feel.

Install pannunga munnadi: pip install customtkinter
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pickle
import numpy as np
import os
import time
import threading
from datetime import datetime
from gtts import gTTS
import customtkinter as ctk
from PIL import Image, ImageTk

MODEL_PATH = "model/sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"
CONFIDENCE_THRESHOLD = 70.0
HOLD_DURATION = 1.5

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: '{MODEL_PATH}' kedaikala. Munnadi 'train_model.py' run pannunga.")
    exit(1)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

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
            # Gradient-ish teal/purple color for a modern look
            cv2.line(frame, points[start_idx], points[end_idx], (200, 160, 60), 3)
        for point in points:
            cv2.circle(frame, point, 5, (255, 255, 255), -1)
            cv2.circle(frame, point, 5, (200, 160, 60), 2)


class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SignSpeak — Sign Language Communication")
        self.root.geometry("1200x700")
        self.root.minsize(1000, 600)

        self.sentence = []
        self.current_word = ""
        self.hold_start_time = None
        self.last_added_word = ""

        self.cap = cv2.VideoCapture(0)

        self.build_ui()
        self.update_frame()

    def build_ui(self):
        # ===== Header =====
        header = ctk.CTkFrame(self.root, height=70, corner_radius=0, fg_color="#1a1a2e")
        header.pack(side="top", fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(header, text="🤟 SignSpeak", font=ctk.CTkFont(size=26, weight="bold"),
                     text_color="#e94560").pack(side="left", padx=25, pady=15)
        ctk.CTkLabel(header, text="Real-time Sign Language to Speech", font=ctk.CTkFont(size=13),
                     text_color="#a0a0a0").pack(side="left", pady=15)

        # ===== Main content area =====
        main_frame = ctk.CTkFrame(self.root, fg_color="#16213e", corner_radius=0)
        main_frame.pack(fill="both", expand=True)

        # ---- Left: Camera panel ----
        left_panel = ctk.CTkFrame(main_frame, fg_color="#0f3460", corner_radius=16)
        left_panel.pack(side="left", fill="both", expand=False, padx=(20, 10), pady=20)

        ctk.CTkLabel(left_panel, text="📷 Live Camera", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="white").pack(anchor="w", padx=15, pady=(15, 5))

        self.video_label = ctk.CTkLabel(left_panel, text="", corner_radius=12)
        self.video_label.pack(padx=15, pady=5)

        self.status_label = ctk.CTkLabel(left_panel, text="Show a sign to the camera...",
                                          font=ctk.CTkFont(size=15, weight="bold"),
                                          text_color="#4ade80")
        self.status_label.pack(pady=(10, 5))

        self.progress = ctk.CTkProgressBar(left_panel, width=480, height=14,
                                            progress_color="#e94560")
        self.progress.set(0)
        self.progress.pack(pady=(0, 15))

        # ---- Right: Sentence + history panel ----
        right_panel = ctk.CTkFrame(main_frame, fg_color="transparent")
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=20)

        # Sentence card
        sentence_card = ctk.CTkFrame(right_panel, fg_color="#0f3460", corner_radius=16)
        sentence_card.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(sentence_card, text="💬 Current Message", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="white").pack(anchor="w", padx=20, pady=(15, 5))

        self.sentence_display = ctk.CTkLabel(sentence_card, text="Start signing...",
                                              font=ctk.CTkFont(size=22, weight="bold"),
                                              text_color="#f9c74f", wraplength=560,
                                              justify="left", anchor="w", height=70)
        self.sentence_display.pack(fill="x", padx=20, pady=(0, 10))

        # Buttons row
        btn_frame = ctk.CTkFrame(sentence_card, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkButton(btn_frame, text="🔊  Speak", font=ctk.CTkFont(size=14, weight="bold"),
                      fg_color="#4ade80", hover_color="#22c55e", text_color="black",
                      corner_radius=10, height=42, command=self.speak_sentence).pack(
            side="left", expand=True, fill="x", padx=(0, 5))

        ctk.CTkButton(btn_frame, text="⌫  Backspace", font=ctk.CTkFont(size=14, weight="bold"),
                      fg_color="#f9c74f", hover_color="#f4a300", text_color="black",
                      corner_radius=10, height=42, command=self.backspace).pack(
            side="left", expand=True, fill="x", padx=5)

        ctk.CTkButton(btn_frame, text="✕  Clear", font=ctk.CTkFont(size=14, weight="bold"),
                      fg_color="#e94560", hover_color="#c9184a", text_color="white",
                      corner_radius=10, height=42, command=self.clear_sentence).pack(
            side="left", expand=True, fill="x", padx=(5, 0))

        # History card
        history_card = ctk.CTkFrame(right_panel, fg_color="#0f3460", corner_radius=16)
        history_card.pack(fill="both", expand=True)

        ctk.CTkLabel(history_card, text="🕓 Communication History", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="white").pack(anchor="w", padx=20, pady=(15, 5))

        self.history_scroll = ctk.CTkScrollableFrame(history_card, fg_color="transparent")
        self.history_scroll.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def add_history_bubble(self, text, timestamp):
        bubble = ctk.CTkFrame(self.history_scroll, fg_color="#e94560", corner_radius=12)
        bubble.pack(fill="x", pady=5, anchor="e")

        ctk.CTkLabel(bubble, text=text, font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="white", wraplength=450, justify="left").pack(
            anchor="w", padx=12, pady=(8, 2))
        ctk.CTkLabel(bubble, text=timestamp, font=ctk.CTkFont(size=10),
                     text_color="#ffd6dd").pack(anchor="e", padx=12, pady=(0, 6))

    def update_frame(self):
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            prediction = None
            confidence = 0.0

            if result.hand_landmarks:
                draw_hand_landmarks(frame, result.hand_landmarks)
                landmark_list = [(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]]
                normalized = normalize_landmarks(landmark_list)
                input_data = np.array(normalized).reshape(1, -1)

                pred = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                conf = max(probabilities) * 100

                if conf >= CONFIDENCE_THRESHOLD:
                    prediction = pred
                    confidence = conf

            self.process_prediction(prediction, confidence)

            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = cv2.resize(display_frame, (500, 380))
            img = Image.fromarray(display_frame)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(500, 380))
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk

        self.root.after(15, self.update_frame)

    def process_prediction(self, prediction, confidence):
        if prediction:
            self.status_label.configure(text=f"✋ Detecting: {prediction}  ({confidence:.0f}%)",
                                         text_color="#4ade80")
            if prediction == self.current_word:
                elapsed = time.time() - self.hold_start_time
                progress_pct = min(elapsed / HOLD_DURATION, 1.0)
                self.progress.set(progress_pct)
                if elapsed >= HOLD_DURATION and prediction != self.last_added_word:
                    self.sentence.append(prediction)
                    self.last_added_word = prediction
                    self.hold_start_time = time.time()
                    self.refresh_sentence_display()
            else:
                self.current_word = prediction
                self.hold_start_time = time.time()
                self.last_added_word = ""
                self.progress.set(0)
        else:
            self.status_label.configure(text="Show a sign to the camera...", text_color="#a0a0a0")
            self.current_word = ""
            self.hold_start_time = None
            self.last_added_word = ""
            self.progress.set(0)

    def refresh_sentence_display(self):
        text = " ".join(self.sentence) if self.sentence else "Start signing..."
        self.sentence_display.configure(text=text)

    def speak_sentence(self):
        if not self.sentence:
            return
        full_text = " ".join(self.sentence)
        timestamp = datetime.now().strftime("%I:%M %p")
        self.add_history_bubble(full_text, timestamp)
        threading.Thread(target=self._speak_thread, args=(full_text,)).start()

    def _speak_thread(self, text):
        try:
            tts = gTTS(text=text, lang='en')
            tts.save("temp_speech.mp3")
            os.system("start temp_speech.mp3")
        except Exception as e:
            print(f"Speech error: {e}")

    def backspace(self):
        if self.sentence:
            self.sentence.pop()
            self.refresh_sentence_display()

    def clear_sentence(self):
        self.sentence = []
        self.refresh_sentence_display()

    def on_close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
