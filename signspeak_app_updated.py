"""
Sign Language Communication App - Dark Gold 3D Edition (Desktop, Two Hands)
================================================================================
NEW: Grammar correction added! When you press SPEAK, the raw signed words
get grammar-corrected (offline, using a local LLM via Ollama) before being
shown in the message/history area. E.g. "my name Prathusha" becomes
"My name is Prathusha".

Install pannunga munnadi:
    pip install customtkinter pillow ollama

Grammar correction setup (one-time, needs Ollama):
    1. Install Ollama: https://ollama.com/download
    2. Run in terminal: ollama pull llama3.2
    3. grammar_correct_offline.py file same folder-la irukanum
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
import customtkinter as ctk
from PIL import Image, ImageTk
from playsound import playsound
import subprocess
import pyttsx3
try:
    import pythoncom
    HAS_PYTHONCOM = True
except ImportError:
    HAS_PYTHONCOM = False
from translations import SIGN_TRANSLATIONS, LANGUAGES, translate_word
from grammar_correct_offline import correct_grammar_offline  # NEW: grammar correction
from neural_tts import speak_neural, MMS_MODEL_MAP  # NEW: high-quality offline Tamil/Indian language voice

ALPHABET_MODEL_PATH = "model/alphabet_model.pkl"

MODEL_PATH = "model/sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_PATH = "face_landmarker.task"
EYEBROW_RAISE_THRESHOLD = 0.4   # blendshape score (0-1) - tune if too sensitive/insensitive
EYEBROW_HOLD_SECONDS = 0.6      # how long eyebrows must stay raised to count as "question"
CONFIDENCE_THRESHOLD = 70.0
ALPHABET_CONFIDENCE_THRESHOLD = 45.0
HOLD_DURATION = 1.2
SIGN_IMAGES_DIR = "sign_images"

BG_BLACK = "#0a0a0a"
CARD_BG = "#161616"
CARD_BG_LIGHT = "#1f1f1f"
GOLD = "#d4af37"
GOLD_BRIGHT = "#f5d576"
GOLD_DIM = "#8a7226"
TEXT_WHITE = "#f5f5f5"
TEXT_GREY = "#9a9a9a"
SHADOW = "#000000"

ctk.set_appearance_mode("dark")

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: '{MODEL_PATH}' kedaikala. Munnadi 'train_model.py' run pannunga.")
    exit(1)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

alphabet_model = None
if os.path.exists(ALPHABET_MODEL_PATH):
    with open(ALPHABET_MODEL_PATH, 'rb') as f:
        alphabet_model = pickle.load(f)

base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    running_mode=vision.RunningMode.IMAGE
)
landmarker = vision.HandLandmarker.create_from_options(options)

# ---- Face Landmarker setup (for eyebrow-raise -> question detection) ----
face_base_options = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    running_mode=vision.RunningMode.IMAGE
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)


def get_eyebrow_raise_score(face_result):
    """Returns 0-1 score for how much the eyebrows are raised (blendshapes)."""
    if not face_result.face_blendshapes:
        return 0.0
    blendshapes = face_result.face_blendshapes[0]
    score = 0.0
    for shape in blendshapes:
        if shape.category_name in ("browInnerUp", "browOuterUpLeft", "browOuterUpRight"):
            score = max(score, shape.score)
    return score


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
            cv2.line(frame, points[start_idx], points[end_idx], (0, 200, 230), 2)
        for point in points:
            cv2.circle(frame, point, 5, (245, 245, 245), -1)
            cv2.circle(frame, point, 5, (0, 200, 230), 2)


class Card3D(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent")
        self.shadow = ctk.CTkFrame(self, fg_color=SHADOW, corner_radius=18)
        self.shadow.place(x=6, y=6, relwidth=1, relheight=1)

        self.card = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=18,
                                  border_width=1, border_color=GOLD_DIM)
        self.card.place(x=0, y=0, relwidth=1, relheight=1)


class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SignSpeak — Gold Edition")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1450, screen_w - 40)
        win_h = min(850, screen_h - 80)
        self.root.geometry(f"{win_w}x{win_h}")
        self.root.minsize(1000, 600)

        try:
            self.root.state('zoomed')
        except Exception:
            try:
                self.root.attributes('-zoomed', True)
            except Exception:
                pass

        self.sentence = []
        self.current_word = ""
        self.hold_start_time = None
        self.last_added_word = ""
        self.current_language = "en"
        self.spell_mode = False
        self.spelled_letters = []
        self.current_letter = ""
        self.letter_hold_start = None
        self.last_added_letter = ""
        self.letter_prediction_history = []
        self.eyebrows_raised_during_sentence = False  # NEW: tracks if a question expression was shown

        self.cap = cv2.VideoCapture(0)

        self.build_ui()
        self.update_frame()

    def build_ui(self):
        self.root.configure(fg_color=BG_BLACK)

        header = ctk.CTkFrame(self.root, height=90, corner_radius=0, fg_color=BG_BLACK)
        header.pack(side="top", fill="x", padx=30, pady=(20, 0))

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left")
        ctk.CTkLabel(title_frame, text="✦ SIGNSPEAK", font=ctk.CTkFont(size=32, weight="bold"),
                     text_color=GOLD).pack(anchor="w")
        ctk.CTkLabel(title_frame, text="SIGN LANGUAGE · SPEECH · COMMUNICATION",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=TEXT_GREY).pack(anchor="w")

        live_badge = ctk.CTkFrame(header, fg_color=CARD_BG, corner_radius=20,
                                   border_width=1, border_color=GOLD)
        live_badge.pack(side="right", pady=10)
        ctk.CTkLabel(live_badge, text="●  LIVE", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=GOLD_BRIGHT).pack(padx=18, pady=8)

        text_to_sign_btn = ctk.CTkButton(
            header, text="⇄  TEXT → SIGN", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=CARD_BG, hover_color=GOLD_DIM, text_color=GOLD,
            border_width=1, border_color=GOLD, corner_radius=20, height=38,
            command=self.open_text_to_sign
        )
        text_to_sign_btn.pack(side="right", padx=(0, 12), pady=10)

        self.spell_btn = ctk.CTkButton(
            header, text="🔤  SPELL MODE: OFF", font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=CARD_BG, hover_color=GOLD_DIM, text_color=GOLD,
            border_width=1, border_color=GOLD, corner_radius=20, height=38,
            command=self.toggle_spell_mode
        )
        self.spell_btn.pack(side="right", padx=(0, 12), pady=10)

        divider = ctk.CTkFrame(self.root, height=2, fg_color=GOLD_DIM)
        divider.pack(fill="x", padx=30, pady=(15, 5))

        body = ctk.CTkFrame(self.root, fg_color=BG_BLACK)
        body.pack(fill="both", expand=True, padx=30, pady=20)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        left_wrap = ctk.CTkFrame(body, fg_color="transparent")
        left_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        cam_card = Card3D(left_wrap)
        cam_card.pack(fill="both", expand=True)
        cam_inner = cam_card.card

        ctk.CTkLabel(cam_inner, text="◆ LIVE CAMERA", font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=GOLD).pack(anchor="w", padx=20, pady=(18, 8))

        video_wrap = ctk.CTkFrame(cam_inner, fg_color=CARD_BG_LIGHT, corner_radius=14,
                                   border_width=1, border_color=GOLD_DIM)
        video_wrap.pack(fill="both", expand=True, padx=20, pady=5)
        self.video_label = ctk.CTkLabel(video_wrap, text="")
        self.video_label.pack(fill="both", expand=True, padx=8, pady=8)

        self.status_label = ctk.CTkLabel(cam_inner, text="Show a sign to the camera...",
                                          font=ctk.CTkFont(size=16, weight="bold"),
                                          text_color=TEXT_GREY)
        self.status_label.pack(pady=(14, 6))

        self.progress = ctk.CTkProgressBar(cam_inner, height=12,
                                            progress_color=GOLD, fg_color=CARD_BG_LIGHT)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=20, pady=(0, 18))

        right_wrap = ctk.CTkFrame(body, fg_color="transparent")
        right_wrap.grid(row=0, column=1, sticky="nsew")

        msg_card_outer = Card3D(right_wrap)
        msg_card_outer.pack(fill="x", pady=(0, 15))
        msg_card_outer.card.pack_propagate(True)
        msg_inner = msg_card_outer.card
        msg_card_outer.configure(height=340)
        msg_card_outer.pack_propagate(False)

        ctk.CTkLabel(msg_inner, text="◆ CURRENT MESSAGE", font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=GOLD).pack(anchor="w", padx=20, pady=(18, 8))

        lang_row = ctk.CTkFrame(msg_inner, fg_color="transparent")
        lang_row.pack(fill="x", padx=20, pady=(0, 10))
        self.lang_buttons = {}
        for idx, (code, label) in enumerate(LANGUAGES.items()):
            row, col = divmod(idx, 3)
            btn = ctk.CTkButton(
                lang_row, text=label, font=ctk.CTkFont(family="Nirmala UI", size=12, weight="bold"),
                width=90, height=30, corner_radius=8,
                fg_color=(GOLD if code == self.current_language else CARD_BG_LIGHT),
                text_color=("#0a0a0a" if code == self.current_language else GOLD),
                border_width=1, border_color=GOLD_DIM,
                hover_color=GOLD_DIM,
                command=lambda c=code: self.set_language(c)
            )
            btn.grid(row=row, column=col, padx=4, pady=4, sticky="ew")
            self.lang_buttons[code] = btn
        lang_row.grid_columnconfigure((0, 1, 2), weight=1)

        self.spell_strip = ctk.CTkFrame(msg_inner, fg_color=CARD_BG, corner_radius=10,
                                         border_width=1, border_color=GOLD_DIM)
        self.spell_buffer_label = ctk.CTkLabel(self.spell_strip, text="Spelling: _",
                                                font=ctk.CTkFont(size=16, weight="bold"),
                                                text_color=GOLD_BRIGHT)
        self.spell_buffer_label.pack(side="left", padx=15, pady=8)
        self.confirm_word_btn = ctk.CTkButton(
            self.spell_strip, text="✓ Confirm Word", font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=GOLD, hover_color=GOLD_BRIGHT, text_color="#0a0a0a",
            corner_radius=8, height=30, command=self.confirm_spelled_word
        )
        self.confirm_word_btn.pack(side="right", padx=10, pady=8)

        self.sentence_display = ctk.CTkLabel(msg_inner, text="Start signing...",
                                              font=ctk.CTkFont(family="Nirmala UI", size=22, weight="bold"),
                                              text_color=GOLD_BRIGHT, wraplength=500,
                                              justify="left", anchor="w", height=50)
        self.sentence_display.pack(fill="x", padx=20)

        self.btn_frame = ctk.CTkFrame(msg_inner, fg_color="transparent")
        btn_frame = self.btn_frame
        btn_frame.pack(fill="x", padx=20, pady=(10, 18))

        ctk.CTkButton(btn_frame, text="🔊 SPEAK", font=ctk.CTkFont(size=13, weight="bold"),
                      fg_color=GOLD, hover_color=GOLD_BRIGHT, text_color="#0a0a0a",
                      corner_radius=10, height=40, command=self.speak_sentence).pack(
            side="left", expand=True, fill="x", padx=(0, 5))

        ctk.CTkButton(btn_frame, text="⌫ UNDO", font=ctk.CTkFont(size=13, weight="bold"),
                      fg_color=CARD_BG_LIGHT, hover_color="#2a2a2a", text_color=GOLD,
                      border_width=1, border_color=GOLD_DIM,
                      corner_radius=10, height=40, command=self.backspace).pack(
            side="left", expand=True, fill="x", padx=5)

        ctk.CTkButton(btn_frame, text="✕ CLEAR", font=ctk.CTkFont(size=13, weight="bold"),
                      fg_color=CARD_BG_LIGHT, hover_color="#2a2a2a", text_color="#e05a5a",
                      border_width=1, border_color="#5a2a2a",
                      corner_radius=10, height=40, command=self.clear_sentence).pack(
            side="left", expand=True, fill="x", padx=(5, 0))

        ctk.CTkLabel(right_wrap, text="◆ COMMUNICATION HISTORY", font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=GOLD).pack(anchor="w", pady=(24, 10))

        self.history_scroll = ctk.CTkScrollableFrame(right_wrap, fg_color="transparent")
        self.history_scroll.pack(fill="both", expand=True)

    def add_history_card(self, text, timestamp):
        wrap = ctk.CTkFrame(self.history_scroll, fg_color="transparent")
        wrap.pack(fill="x", pady=6)

        shadow = ctk.CTkFrame(wrap, fg_color=SHADOW, corner_radius=14)
        shadow.place(x=4, y=4, relwidth=1, relheight=1)

        card = ctk.CTkFrame(wrap, fg_color=CARD_BG_LIGHT, corner_radius=14,
                             border_width=1, border_color=GOLD_DIM, height=70)
        card.pack(fill="x")
        card.pack_propagate(False)

        strip = ctk.CTkFrame(card, fg_color=GOLD, width=6, corner_radius=0)
        strip.pack(side="left", fill="y", padx=(0, 12), pady=8)

        text_col = ctk.CTkFrame(card, fg_color="transparent")
        text_col.pack(side="left", fill="both", expand=True, pady=10)

        ctk.CTkLabel(text_col, text=text, font=ctk.CTkFont(family="Nirmala UI", size=15, weight="bold"),
                     text_color=TEXT_WHITE, anchor="w", justify="left").pack(anchor="w")
        ctk.CTkLabel(text_col, text=timestamp, font=ctk.CTkFont(size=11),
                     text_color=GOLD_DIM, anchor="w").pack(anchor="w")

    def open_text_to_sign(self):
        popup = ctk.CTkToplevel(self.root)
        popup.title("Text → Sign Language")
        popup.geometry("700x600")
        popup.configure(fg_color=BG_BLACK)
        popup.attributes("-topmost", True)

        ctk.CTkLabel(popup, text="⇄  Text → Sign Language",
                     font=ctk.CTkFont(size=22, weight="bold"), text_color=GOLD).pack(pady=(20, 5))
        ctk.CTkLabel(popup, text="Type a word or sentence, andha sign eppadi pannuradhu nu paarunga",
                     font=ctk.CTkFont(size=12), text_color=TEXT_GREY).pack(pady=(0, 15))

        entry_frame = ctk.CTkFrame(popup, fg_color="transparent")
        entry_frame.pack(fill="x", padx=25, pady=(0, 10))

        entry = ctk.CTkEntry(entry_frame, placeholder_text="e.g. Hello Thank You",
                              font=ctk.CTkFont(family="Nirmala UI", size=15),
                              height=40, fg_color=CARD_BG_LIGHT, border_color=GOLD_DIM,
                              text_color=TEXT_WHITE)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        result_scroll = ctk.CTkScrollableFrame(popup, fg_color="transparent")
        result_scroll.pack(fill="both", expand=True, padx=25, pady=(10, 20))

        def show_signs():
            for widget in result_scroll.winfo_children():
                widget.destroy()

            typed_text = entry.get().strip().lower()
            if not typed_text:
                return

            vocab_sorted = sorted(SIGN_TRANSLATIONS.keys(), key=lambda w: -len(w))
            matched = []
            remaining = typed_text
            for word in vocab_sorted:
                if word.lower() in remaining:
                    matched.append(word)
                    remaining = remaining.replace(word.lower(), " ")

            if not matched:
                ctk.CTkLabel(result_scroll, text="No matching sign found in vocabulary.",
                             font=ctk.CTkFont(size=14), text_color="#e05a5a").pack(pady=20)
                return

            for word in matched:
                self._add_sign_result_card(result_scroll, word)

        entry.bind("<Return>", lambda e: show_signs())

        show_btn = ctk.CTkButton(entry_frame, text="Show Signs", font=ctk.CTkFont(size=13, weight="bold"),
                                  fg_color=GOLD, hover_color=GOLD_BRIGHT, text_color="#0a0a0a",
                                  corner_radius=10, height=40, width=110, command=show_signs)
        show_btn.pack(side="left")

    def _add_sign_result_card(self, parent, word):
        safe_name = word.replace(" ", "_")
        img_path = os.path.join(SIGN_IMAGES_DIR, f"{safe_name}.png")

        card = ctk.CTkFrame(parent, fg_color=CARD_BG_LIGHT, corner_radius=14,
                             border_width=1, border_color=GOLD_DIM)
        card.pack(fill="x", pady=8)

        if os.path.exists(img_path):
            pil_img = Image.open(img_path)
            pil_img.thumbnail((220, 220))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            img_label = ctk.CTkLabel(card, text="", image=ctk_img)
            img_label.pack(side="left", padx=15, pady=15)
        else:
            placeholder = ctk.CTkFrame(card, fg_color=CARD_BG, width=150, height=150, corner_radius=10)
            placeholder.pack(side="left", padx=15, pady=15)
            placeholder.pack_propagate(False)
            ctk.CTkLabel(placeholder, text="No photo\nyet", font=ctk.CTkFont(size=12),
                         text_color=TEXT_GREY).pack(expand=True)

        ctk.CTkLabel(card, text=word, font=ctk.CTkFont(family="Nirmala UI", size=18, weight="bold"),
                     text_color=GOLD_BRIGHT).pack(side="left", padx=10, anchor="center")

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
                features, detected = extract_two_hand_features(result)
                if detected:
                    input_data = np.array(features).reshape(1, -1)
                    active_model = alphabet_model if (self.spell_mode and alphabet_model is not None) else model
                    active_threshold = ALPHABET_CONFIDENCE_THRESHOLD if self.spell_mode else CONFIDENCE_THRESHOLD
                    pred = active_model.predict(input_data)[0]
                    probabilities = active_model.predict_proba(input_data)[0]
                    conf = max(probabilities) * 100
                    if self.spell_mode:
                        print(f"[DEBUG] Letter guess: {pred}  confidence: {conf:.1f}%")
                    if conf >= active_threshold:
                        prediction = pred
                        confidence = conf

            self.process_prediction(prediction, confidence)

            # ---- NEW: Face expression detection (eyebrow raise -> question) ----
            face_result = face_landmarker.detect(mp_image)
            eyebrow_score = get_eyebrow_raise_score(face_result)
            if eyebrow_score > 0.4:  # threshold - tweak if too sensitive/insensitive
                self.eyebrows_raised_during_sentence = True
                self.status_label.configure(text="🤨 Question expression detected", text_color=GOLD_BRIGHT)

            self.video_label.update_idletasks()
            target_w = self.video_label.winfo_width()
            target_h = self.video_label.winfo_height()
            if target_w < 50 or target_h < 50:
                target_w, target_h = 640, 480

            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = cv2.resize(display_frame, (target_w, target_h))
            img = Image.fromarray(display_frame)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(target_w, target_h))
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk

        self.root.after(15, self.update_frame)

    def toggle_spell_mode(self):
        if alphabet_model is None:
            self.status_label.configure(
                text="Alphabet model illa! 'train_alphabet_model.py' run pannunga.",
                text_color="#e05a5a")
            return
        self.spell_mode = not self.spell_mode
        self.spelled_letters = []
        self.current_letter = ""
        self.letter_hold_start = None
        self.last_added_letter = ""
        self.letter_prediction_history = []
        if self.spell_mode:
            self.spell_btn.configure(text="🔤  SPELL MODE: ON", fg_color=GOLD, text_color="#0a0a0a")
            self.spell_strip.pack(fill="x", padx=20, pady=(0, 10), before=self.btn_frame)
            self.refresh_spell_buffer()
        else:
            self.spell_btn.configure(text="🔤  SPELL MODE: OFF", fg_color=CARD_BG, text_color=GOLD)
            self.spell_strip.pack_forget()

    def refresh_spell_buffer(self):
        text = "".join(self.spelled_letters) if self.spelled_letters else "_"
        self.spell_buffer_label.configure(text=f"Spelling: {text}")

    def confirm_spelled_word(self):
        if not self.spelled_letters:
            return
        word = "".join(self.spelled_letters).capitalize()
        self.sentence.append(word)
        self.spelled_letters = []
        self.refresh_spell_buffer()
        self.refresh_sentence_display()

    def process_prediction(self, prediction, confidence):
        if self.spell_mode:
            self._process_letter_prediction(prediction, confidence)
            return

        if prediction:
            self.status_label.configure(text=f"✋  {prediction}   ·   {confidence:.0f}%",
                                         text_color=GOLD_BRIGHT)
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
            self.status_label.configure(text="Show a sign to the camera...", text_color=TEXT_GREY)
            self.current_word = ""
            self.hold_start_time = None
            self.last_added_word = ""
            self.progress.set(0)

    def _process_letter_prediction(self, prediction, confidence):
        LETTER_HOLD = 0.9

        self.letter_prediction_history.append(prediction)
        if len(self.letter_prediction_history) > 5:
            self.letter_prediction_history.pop(0)

        non_none = [p for p in self.letter_prediction_history if p is not None]
        if len(non_none) >= 3:
            smoothed_prediction = max(set(non_none), key=non_none.count)
        else:
            smoothed_prediction = None
        prediction = smoothed_prediction

        if prediction:
            self.status_label.configure(text=f"🔤  {prediction}   ·   {confidence:.0f}%",
                                         text_color=GOLD_BRIGHT)
            if prediction == self.current_letter:
                elapsed = time.time() - self.letter_hold_start
                progress_pct = min(elapsed / LETTER_HOLD, 1.0)
                self.progress.set(progress_pct)
                if elapsed >= LETTER_HOLD and prediction != self.last_added_letter:
                    self.spelled_letters.append(prediction)
                    self.last_added_letter = prediction
                    self.letter_hold_start = time.time()
                    self.refresh_spell_buffer()
                    print(f"[SPELL] Letter added: {prediction}  ->  buffer: {''.join(self.spelled_letters)}")
            else:
                self.current_letter = prediction
                self.letter_hold_start = time.time()
                self.last_added_letter = ""
                self.progress.set(0)
        else:
            self.status_label.configure(text="Spell mode: show a letter...", text_color=TEXT_GREY)
            self.current_letter = ""
            self.letter_hold_start = None
            self.last_added_letter = ""
            self.progress.set(0)

    def set_language(self, lang_code):
        self.current_language = lang_code
        for code, btn in self.lang_buttons.items():
            if code == lang_code:
                btn.configure(fg_color=GOLD, text_color="#0a0a0a")
            else:
                btn.configure(fg_color=CARD_BG_LIGHT, text_color=GOLD)
        self.refresh_sentence_display()

    def refresh_sentence_display(self):
        if self.sentence:
            translated_words = [translate_word(w, self.current_language) for w in self.sentence]
            text = " ".join(translated_words)
        else:
            text = "Start signing..."
        self.sentence_display.configure(text=text)

    # ---------------------------------------------------------------
    # SPEAK flow — NOW WITH GRAMMAR CORRECTION
    # ---------------------------------------------------------------
    def speak_sentence(self):
        """
        Called when SPEAK button is pressed. Instead of speaking the raw
        signed words directly, we first send them through offline grammar
        correction (runs in a background thread so the camera/UI never
        freezes), then display + speak the corrected sentence.
        """
        if not self.sentence:
            return

        translated_words = [translate_word(w, self.current_language) for w in self.sentence]
        raw_text = " ".join(translated_words)

        # NEW: if eyebrows were raised at any point while signing this sentence,
        # treat it as a question (real ISL grammar uses facial expression, not
        # just hand signs, to mark questions).
        if self.eyebrows_raised_during_sentence and not raw_text.strip().endswith("?"):
            raw_text += "?"
        self.eyebrows_raised_during_sentence = False  # reset for the next sentence

        self.status_label.configure(text="Fixing grammar...", text_color=GOLD_DIM)

        threading.Thread(
            target=self._correct_and_speak,
            args=(raw_text, list(self.sentence), self.current_language)
        ).start()

    def _correct_and_speak(self, raw_text, original_words, lang_code):
        """Runs in a background thread — calls the local LLM to fix grammar."""
        try:
            result = correct_grammar_offline(raw_text)
            corrected_text = result["corrected"]
        except Exception as e:
            print(f"Grammar correction error: {e}")
            corrected_text = raw_text  # fallback: use raw text if correction fails/Ollama not running

        try:
            if self.root.winfo_exists():
                self.root.after(0, lambda: self._finish_speak(corrected_text, original_words, lang_code))
        except RuntimeError:
            print("Window closed before grammar correction finished — skipping UI update.")

    def _finish_speak(self, corrected_text, original_words, lang_code):
        """Runs on the main thread — updates the UI and starts speaking."""
        timestamp = datetime.now().strftime("%I:%M %p")
        self.add_history_card(corrected_text, timestamp)
        self.sentence_display.configure(text=corrected_text)
        self.status_label.configure(text="Show a sign to the camera...", text_color=TEXT_GREY)

        # Speak the FULL grammar-corrected sentence as one sentence.
        # English -> Windows built-in voice (better quality).
        # Other languages (Tamil, Hindi, etc.) -> eSpeak NG (offline, supports
        # many languages Windows doesn't have a built-in voice for).
        threading.Thread(target=self._speak_multilang, args=(corrected_text, lang_code)).start()

    # Map your app's language codes to eSpeak NG voice codes.
    # Adjust the keys here to match whatever codes are used in LANGUAGES /
    # translations.py in your project (check translations.py if unsure).
    ESPEAK_VOICE_MAP = {
        "ta": "ta",   # Tamil
        "hi": "hi",   # Hindi
        "te": "te",   # Telugu
        "kn": "kn",   # Kannada
        "ml": "ml",   # Malayalam
        "bn": "bn",   # Bengali
        "mr": "mr",   # Marathi
        "gu": "gu",   # Gujarati
        "pa": "pa",   # Punjabi
    }

    ESPEAK_PATH = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

    def _speak_multilang(self, text, lang_code):
        if lang_code == "en":
            # English -> Windows built-in voice
            self._speak_offline_windows(text)
        elif lang_code in MMS_MODEL_MAP:
            # Tamil/Hindi/Telugu/Kannada/Malayalam -> neural voice (much better
            # quality than eSpeak). First use per language downloads the model
            # once, then it's cached and fully offline after that.
            speak_neural(text, lang_code)
        else:
            # Fallback for any language without a neural model configured
            self._speak_offline_windows(text)

    def _speak_espeak(self, text, voice_code):
        import tempfile
        # Windows mangles Unicode (Tamil/Hindi/etc) text passed directly as a
        # command-line argument. Writing to a UTF-8 temp file and using
        # espeak-ng's -f flag avoids that encoding problem entirely.
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(text)
                temp_path = f.name

            creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            result = subprocess.run(
                [self.ESPEAK_PATH, "-v", voice_code, "-f", temp_path],
                creationflags=creationflags,
                timeout=15,
                capture_output=True,
                text=True,
            )
            print(f"[eSpeak] voice={voice_code} text='{text}' returncode={result.returncode} stderr={result.stderr}")

            os.remove(temp_path)
        except Exception as e:
            print(f"eSpeak NG error for '{text}' (voice={voice_code}): {e}")

    def _speak_thread(self, words, lang_code):
        for word in words:
            if word in SIGN_TRANSLATIONS:
                safe_name = word.replace(" ", "_")
                file_path = os.path.join("audio", lang_code, f"{safe_name}.mp3")
                if os.path.exists(file_path):
                    try:
                        playsound(file_path)
                    except Exception as e:
                        print(f"Playback error for '{word}': {e}")
                else:
                    print(f"Audio file missing: {file_path}")
            else:
                self._speak_offline_windows(word.title())

    @staticmethod
    def _speak_offline_windows(text):
        safe_text = text.replace('"', "").replace("`", "").replace("$", "")
        ps_command = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Rate = -1; "
            f'$s.Speak("{safe_text}");'
        )
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_command],
                creationflags=creationflags,
                timeout=15,
            )
        except Exception as e:
            print(f"Offline TTS (PowerShell) error for '{text}': {e}")

    def backspace(self):
        if self.sentence:
            self.sentence.pop()
            self.refresh_sentence_display()

    def clear_sentence(self):
        self.sentence = []
        self.eyebrows_raised_during_sentence = False
        self.refresh_sentence_display()

    def on_close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
