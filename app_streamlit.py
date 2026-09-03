"""
Sign Language Communication App - Streamlit Web Version
=============================================================
Idhu app ah browser la run aagum - laptop or phone, edhu venalum,
oru URL open pannina, camera access pannitu sign language detect pannum.

Run pannuradhu:
    streamlit run app_streamlit.py

(streamlit run command mattum use pannunga, 'python app_streamlit.py' illa)
"""

import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pickle
import numpy as np
import os
import time
from datetime import datetime
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import queue

# ============ Page config ============
st.set_page_config(page_title="SignSpeak", page_icon="🤟", layout="wide")

MODEL_PATH = "model/sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"
CONFIDENCE_THRESHOLD = 70.0
HOLD_DURATION = 1.0

# ============ Custom CSS - rich, modern, polished look ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 15% 15%, #1e2a5e 0%, #0d1128 45%, #05060f 100%);
    }

    /* Hide default streamlit chrome for a cleaner app feel */
    #MainMenu, footer, header { visibility: hidden; }

    /* ===== Hero header ===== */
    .hero-wrap {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 28px;
        border-radius: 20px;
        background: linear-gradient(120deg, rgba(233,69,96,0.18), rgba(15,52,96,0.35));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 22px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    .hero-title {
        font-size: 34px;
        font-weight: 800;
        background: linear-gradient(90deg, #ff6b81, #f9c74f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-sub {
        color: #b8bfd6;
        font-size: 14px;
        font-weight: 500;
        margin-top: 2px;
    }
    .live-chip {
        background: rgba(74, 222, 128, 0.15);
        border: 1px solid #4ade80;
        color: #4ade80;
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .live-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #4ade80;
        margin-right: 6px;
        animation: pulse 1.4s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(74,222,128,0.6); }
        70% { box-shadow: 0 0 0 8px rgba(74,222,128,0); }
        100% { box-shadow: 0 0 0 0 rgba(74,222,128,0); }
    }

    /* ===== Stat chips row ===== */
    .stat-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 14px 18px;
        text-align: center;
        backdrop-filter: blur(6px);
    }
    .stat-value { font-size: 24px; font-weight: 800; color: #f9c74f; }
    .stat-label { font-size: 11px; color: #9aa3c0; text-transform: uppercase; letter-spacing: 1px; }

    /* ===== Glass section cards ===== */
    .glass-card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 20px;
        padding: 20px 22px;
        margin-bottom: 18px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }
    .glass-card h4 {
        margin-top: 0;
        color: #ffffff;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        opacity: 0.85;
    }

    .sentence-card {
        background: linear-gradient(135deg, rgba(249,199,79,0.10), rgba(233,69,96,0.10));
        border: 1px solid rgba(249,199,79,0.25);
        padding: 26px;
        border-radius: 18px;
        margin-bottom: 16px;
        min-height: 90px;
        display: flex;
        align-items: center;
    }
    .sentence-text {
        font-size: 30px;
        font-weight: 700;
        color: #f9c74f;
        line-height: 1.3;
    }

    .history-bubble {
        background: linear-gradient(135deg, #e94560, #c9184a);
        color: white;
        padding: 12px 18px;
        border-radius: 16px 16px 4px 16px;
        margin-bottom: 10px;
        text-align: right;
        box-shadow: 0 4px 14px rgba(233,69,96,0.35);
        font-size: 15px;
        font-weight: 600;
    }
    .timestamp {
        font-size: 11px;
        color: #ffd6dd;
        font-weight: 400;
        display: block;
        margin-top: 3px;
    }

    /* ===== Buttons ===== */
    .stButton>button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 10px 0 !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.35);
    }

    /* Section labels */
    .section-label {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #9aa3c0;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============ Load model (cache pannurom, ovvoru rerun ku reload aagama irukka) ============
@st.cache_resource
def load_resources():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        running_mode=vision.RunningMode.IMAGE
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    return model, landmarker

model, landmarker = load_resources()


def normalize_single_hand(landmarks):
    """Oru kai oda 21 landmarks ah, wrist (point 0) ku relative ah normalize pannurom."""
    base_x, base_y = landmarks[0].x, landmarks[0].y
    normalized = []
    for lm in landmarks:
        normalized.extend([lm.x - base_x, lm.y - base_y, lm.z])
    return normalized


def extract_two_hand_features(result):
    """
    Left hand -> first 63 values, Right hand -> next 63 values.
    Kai illana andha part zeros ah irukkum. Total 126 features.
    """
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
            cv2.line(frame, points[start_idx], points[end_idx], (60, 160, 200), 3)
        for point in points:
            cv2.circle(frame, point, 5, (255, 255, 255), -1)
            cv2.circle(frame, point, 5, (60, 160, 200), 2)


# ============ Session state initialize pannurom ============
if "sentence" not in st.session_state:
    st.session_state.sentence = []
if "history" not in st.session_state:
    st.session_state.history = []
if "current_word" not in st.session_state:
    st.session_state.current_word = ""
if "hold_start_time" not in st.session_state:
    st.session_state.hold_start_time = None
if "last_added_word" not in st.session_state:
    st.session_state.last_added_word = ""

result_queue = queue.Queue(maxsize=1)


frame_counter = {"count": 0}
last_result = {"prediction": None, "confidence": 0.0}


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    # Speed optimization: process mattum ovvoru 2nd frame, matha frames la
    # last known result ah reuse pannurom - lag kammi aagum
    frame_counter["count"] += 1
    process_this_frame = (frame_counter["count"] % 2 == 0)

    prediction = last_result["prediction"]
    confidence = last_result["confidence"]

    if process_this_frame:
        # Detection ku chinna resolution use pannurom - fast aagum
        small = cv2.resize(img, (320, 240))
        rgb_frame = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        prediction = None
        confidence = 0.0

        if result.hand_landmarks:
            draw_hand_landmarks(img, result.hand_landmarks)
            features, detected = extract_two_hand_features(result)

            if detected:
                input_data = np.array(features).reshape(1, -1)
                pred = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                conf = max(probabilities) * 100

                if conf >= CONFIDENCE_THRESHOLD:
                    prediction = pred
                    confidence = conf

        last_result["prediction"] = prediction
        last_result["confidence"] = confidence
    else:
        # Skip pannina frame la kooda, last detected hand landmarks draw pannurom (visual continuity ku)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)
        if result.hand_landmarks:
            draw_hand_landmarks(img, result.hand_landmarks)

    # Prediction ah main thread ku pass pannurom queue mூlam
    if not result_queue.empty():
        try:
            result_queue.get_nowait()
        except queue.Empty:
            pass
    result_queue.put({"prediction": prediction, "confidence": confidence})

    return av.VideoFrame.from_ndarray(img, format="bgr24")


def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("temp_speech.mp3")
        os.system("start temp_speech.mp3")
    except Exception as e:
        print(f"Speech error: {e}")


# ============ Sidebar - vocabulary & how-to ============
with st.sidebar:
    st.markdown("## 🤟 SignSpeak")
    st.caption("Sign Language → Speech, in real time")
    st.markdown("---")
    st.markdown("#### 📖 How to use")
    st.markdown(
        "1. Allow camera access\n"
        "2. Show a sign, hold for **1.5 sec**\n"
        "3. Word gets added to your message\n"
        "4. Tap **🔊 Speak** to convert to voice"
    )
    st.markdown("---")
    st.markdown("#### 🗂️ Supported Signs")
    vocab = sorted(list(getattr(model, "classes_", [])))
    if vocab:
        st.markdown(" · ".join([f"`{v}`" for v in vocab]))
    st.markdown("---")
    st.caption("Built by Byte Squad · WEC Hackathon 2026")

# ============ Hero header ============
st.markdown(f"""
<div class="hero-wrap">
    <div>
        <div class="hero-title">🤟 SignSpeak</div>
        <div class="hero-sub">Real-time Sign Language to Speech Communication</div>
    </div>
    <div class="live-chip"><span class="live-dot"></span>LIVE</div>
</div>
""", unsafe_allow_html=True)

# ============ Stats row ============
vocab_count = len(getattr(model, "classes_", []))
stat_c1, stat_c2, stat_c3 = st.columns(3)
with stat_c1:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{vocab_count}</div>'
                f'<div class="stat-label">Signs Supported</div></div>', unsafe_allow_html=True)
with stat_c2:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{len(st.session_state.history)}</div>'
                f'<div class="stat-label">Messages Spoken</div></div>', unsafe_allow_html=True)
with stat_c3:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{len(st.session_state.sentence)}</div>'
                f'<div class="stat-label">Words in Draft</div></div>', unsafe_allow_html=True)

st.write("")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h4>📷 Live Camera</h4>", unsafe_allow_html=True)
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(
        key="sign-language",
        video_frame_callback=video_frame_callback,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": {"width": {"ideal": 480}, "height": {"ideal": 360}, "frameRate": {"ideal": 15}},
            "audio": False
        },
    )
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    status_placeholder.info("Show a sign to the camera...")
    progress_placeholder.progress(0)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h4>💬 Current Message</h4>", unsafe_allow_html=True)
    sentence_placeholder = st.empty()

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    speak_clicked = btn_col1.button("🔊 Speak", use_container_width=True)
    backspace_clicked = btn_col2.button("⌫ Undo", use_container_width=True)
    clear_clicked = btn_col3.button("✕ Clear", use_container_width=True)

    if backspace_clicked and st.session_state.sentence:
        st.session_state.sentence.pop()
    if clear_clicked:
        st.session_state.sentence = []
    if speak_clicked and st.session_state.sentence:
        full_text = " ".join(st.session_state.sentence)
        timestamp = datetime.now().strftime("%I:%M %p")
        st.session_state.history.insert(0, {"text": full_text, "time": timestamp})
        speak_text(full_text)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h4>🕓 Communication History</h4>", unsafe_allow_html=True)
    history_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)


def render_sentence():
    text = " ".join(st.session_state.sentence) if st.session_state.sentence else "Start signing..."
    sentence_placeholder.markdown(
        f'<div class="sentence-card"><span class="sentence-text">{text}</span></div>',
        unsafe_allow_html=True
    )


def render_history():
    if not st.session_state.history:
        history_placeholder.info("No messages yet.")
        return
    html = ""
    for item in st.session_state.history[:15]:
        html += f'<div class="history-bubble">{item["text"]}<br><span class="timestamp">{item["time"]}</span></div>'
    history_placeholder.markdown(html, unsafe_allow_html=True)


render_sentence()
render_history()

# ============ Live update fragment - auto-refreshes without blocking or getting stuck ============
@st.fragment(run_every=0.05)
def live_update():
    if not webrtc_ctx.state.playing:
        status_placeholder.info("Camera off. Click START to begin.")
        progress_placeholder.progress(0)
        return

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        return  # idhu frame la edhum puthusa illa, next 0.1s la try pannum

    prediction = result["prediction"]
    confidence = result["confidence"]

    if prediction:
        status_placeholder.success(f"✋ Detecting: **{prediction}**  ·  {confidence:.0f}% confident")
        if prediction == st.session_state.current_word:
            elapsed = time.time() - st.session_state.hold_start_time
            progress_pct = min(elapsed / HOLD_DURATION, 1.0)
            progress_placeholder.progress(progress_pct)
            if elapsed >= HOLD_DURATION and prediction != st.session_state.last_added_word:
                st.session_state.sentence.append(prediction)
                st.session_state.last_added_word = prediction
                st.session_state.hold_start_time = time.time()
                render_sentence()
        else:
            st.session_state.current_word = prediction
            st.session_state.hold_start_time = time.time()
            st.session_state.last_added_word = ""
            progress_placeholder.progress(0)
    else:
        status_placeholder.info("Show a sign to the camera...")
        st.session_state.current_word = ""
        st.session_state.hold_start_time = None
        st.session_state.last_added_word = ""
        progress_placeholder.progress(0)


live_update()
