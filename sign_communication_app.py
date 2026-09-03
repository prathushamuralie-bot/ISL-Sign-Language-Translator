"""
Sign Language Communication App
====================================
Idhu oru full GUI app - kai sign pannina, adha identify pannitu, sentence build pannitu,
voice ah convert pannitu, communication history um vachukum.

Idhu "communication channel" mathiri - oru pakkam sign pannuranga,
adhu detect aagi, screen la text ah kaanum + voice ah sollum.

Requirements: pip install pillow (already irukka nu check pannunga)
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
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

MODEL_PATH = "model/sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"
CONFIDENCE_THRESHOLD = 70.0
HOLD_DURATION = 1.5

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
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)


class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Communication App")
        self.root.geometry("1100x650")
        self.root.configure(bg="#1e1e2e")

        self.sentence = []
        self.current_word = ""
        self.hold_start_time = None
        self.last_added_word = ""

        self.cap = cv2.VideoCapture(0)

        self.build_ui()
        self.update_frame()

    def build_ui(self):
        # ===== Left: Camera feed =====
        left_frame = tk.Frame(self.root, bg="#1e1e2e")
        left_frame.pack(side="left", padx=10, pady=10)

        self.video_label = tk.Label(left_frame, bg="black")
        self.video_label.pack()

        self.status_label = tk.Label(left_frame, text="Show a sign...",
                                      font=("Arial", 14, "bold"), fg="#a6e3a1", bg="#1e1e2e")
        self.status_label.pack(pady=5)

        self.progress = ttk.Progressbar(left_frame, length=400, mode="determinate")
        self.progress.pack(pady=5)

        # ===== Right: Sentence + controls + history =====
        right_frame = tk.Frame(self.root, bg="#1e1e2e")
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        tk.Label(right_frame, text="Current Sentence", font=("Arial", 16, "bold"),
                 fg="white", bg="#1e1e2e").pack(anchor="w")

        self.sentence_display = tk.Label(right_frame, text="(empty)", font=("Arial", 20),
                                          fg="#f9e2af", bg="#313244", wraplength=500,
                                          justify="left", anchor="w", width=35, height=3)
        self.sentence_display.pack(pady=10, fill="x")

        # Buttons
        button_frame = tk.Frame(right_frame, bg="#1e1e2e")
        button_frame.pack(pady=10, fill="x")

        speak_btn = tk.Button(button_frame, text="🔊 Speak", font=("Arial", 12, "bold"),
                               bg="#a6e3a1", fg="black", command=self.speak_sentence, width=12)
        speak_btn.grid(row=0, column=0, padx=5)

        back_btn = tk.Button(button_frame, text="⌫ Backspace", font=("Arial", 12, "bold"),
                              bg="#f9e2af", fg="black", command=self.backspace, width=12)
        back_btn.grid(row=0, column=1, padx=5)

        clear_btn = tk.Button(button_frame, text="✕ Clear", font=("Arial", 12, "bold"),
                               bg="#f38ba8", fg="black", command=self.clear_sentence, width=12)
        clear_btn.grid(row=0, column=2, padx=5)

        # History
        tk.Label(right_frame, text="Communication History", font=("Arial", 16, "bold"),
                 fg="white", bg="#1e1e2e").pack(anchor="w", pady=(20, 5))

        history_frame = tk.Frame(right_frame, bg="#1e1e2e")
        history_frame.pack(fill="both", expand=True)

        self.history_listbox = tk.Listbox(history_frame, font=("Arial", 12), bg="#313244",
                                           fg="white", height=12, selectbackground="#585b70")
        self.history_listbox.pack(fill="both", expand=True, side="left")

        scrollbar = tk.Scrollbar(history_frame, command=self.history_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.history_listbox.config(yscrollcommand=scrollbar.set)

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

            # Frame ah Tkinter la kaatanum, RGB ku convert pannurom
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = cv2.resize(display_frame, (560, 420))
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(15, self.update_frame)

    def process_prediction(self, prediction, confidence):
        if prediction:
            self.status_label.config(text=f"Detecting: {prediction} ({confidence:.0f}%)")
            if prediction == self.current_word:
                elapsed = time.time() - self.hold_start_time
                progress_pct = min((elapsed / HOLD_DURATION) * 100, 100)
                self.progress['value'] = progress_pct
                if elapsed >= HOLD_DURATION and prediction != self.last_added_word:
                    self.sentence.append(prediction)
                    self.last_added_word = prediction
                    self.hold_start_time = time.time()
                    self.refresh_sentence_display()
            else:
                self.current_word = prediction
                self.hold_start_time = time.time()
                self.last_added_word = ""
                self.progress['value'] = 0
        else:
            self.status_label.config(text="Show a sign...")
            self.current_word = ""
            self.hold_start_time = None
            self.last_added_word = ""
            self.progress['value'] = 0

    def refresh_sentence_display(self):
        text = " ".join(self.sentence) if self.sentence else "(empty)"
        self.sentence_display.config(text=text)

    def speak_sentence(self):
        if not self.sentence:
            return
        full_text = " ".join(self.sentence)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_listbox.insert(0, f"[{timestamp}] {full_text}")
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
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
