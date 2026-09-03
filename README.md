# 🤟 ISL Sign Language Translator

An AI-powered **Indian Sign Language (ISL) Translator** designed to bridge the communication gap between deaf and hearing communities.

The system recognizes hand signs and converts them into meaningful text and speech. It also explores **facial expressions and grammatical markers**, which are an important part of sign-language communication.

---

## 🚀 Key Features

### 🤟 Sign Language Recognition

* Recognizes Indian Sign Language hand signs.
* Supports alphabet/sign recognition.
* Supports one-hand and two-hand signs.
* Real-time camera-based recognition.

### 😊 Facial Expression Recognition

Facial expressions can provide additional grammatical and contextual information in sign language.

The project integrates **MediaPipe Face Landmarker** to detect facial movements and expressions.

Example:

**Eyebrow Raise → Question / Interrogative Marker ❓**

This helps the translator understand not only the hand sign but also the facial context.

### 🗣️ Text-to-Speech

Recognized signs are converted into speech so that hearing users can understand the communication.

### 🌐 Multilingual Audio Support

The application provides audio support for multiple Indian languages.

### 🧠 Machine Learning

The project uses trained machine-learning models for sign recognition and MediaPipe landmark detection for hand and face analysis.

---

## 💡 Problem Statement

Most sign-language recognition systems primarily focus on **hand gestures**.

However, sign languages also use **facial expressions and non-manual markers** to convey grammatical information, emotions, and sentence meaning.

Therefore, a translator that considers both **hand signs + facial expressions** can provide more meaningful communication.

---

## 💻 Technologies Used

* Python
* OpenCV
* MediaPipe
* Scikit-learn
* Streamlit
* Machine Learning
* Text-to-Speech
* Git & GitHub

---

## 🏗️ System Workflow

```text
        Camera Input
             ↓
     ┌───────────────┐
     │ Hand Detection│
     └───────┬───────┘
             ↓
      Hand Landmarks
             ↓
      Sign Recognition
             ↓
     ┌─────────────────┐
     │ Facial Analysis │
     └────────┬────────┘
              ↓
    Facial / Grammar Marker
              ↓
       Context Processing
              ↓
        Text Generation
              ↓
       Text-to-Speech
              ↓
          🔊 Speech
```

---

## 📁 Project Structure

```text
ISL-Sign-Language-Translator/
│
├── app.py
├── app_streamlit.py
├── main.py
│
├── data_collection.py
├── data_collection_alphabet.py
├── data_collection_auto.py
├── data_collection_two_hands.py
│
├── train_model.py
├── train_alphabet_model.py
│
├── grammar_correct.py
├── grammar_correct_offline.py
│
├── generate_audio.py
├── neural_tts.py
├── translations.py
│
├── face_landmarker.task
├── hand_landmarker.task
│
├── dataset/
│   ├── alphabet_data.csv
│   ├── sign_data.csv
│   └── sign_data_two_hands.csv
│
├── model/
│   ├── alphabet_model.pkl
│   └── sign_model.pkl
│
└── audio/
    ├── en/
    ├── hi/
    ├── ta/
    ├── te/
    ├── kn/
    └── ml/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/prathushamuralie-bot/ISL-Sign-Language-Translator.git
```

Move into the project:

```bash
cd ISL-Sign-Language-Translator
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

If using Streamlit:

```bash
streamlit run app_streamlit.py
```

or run the required Python application from PyCharm.

---

## 🎯 Real-World Impact

The project aims to make communication more accessible for people who use Indian Sign Language.

Potential applications include:

* 🏫 Educational institutions
* 🏥 Hospitals
* 🏛️ Government offices
* 🏢 Workplaces
* 🛒 Public service environments
* 👥 Everyday communication

The long-term goal is to build a more inclusive communication interface where **signs, facial expressions and grammatical markers** are interpreted together.

---

## 🔬 Innovation

### Traditional Approach

```text
Hand Sign → Text
```

### Proposed Approach

```text
Hand Sign
    +
Facial Expression
    +
Grammatical Marker
    ↓
Context-Aware Interpretation
    ↓
Text
    ↓
Speech
```

This makes the system more context-aware instead of relying only on hand gestures.

---

## 🔮 Future Scope

* Expand the ISL vocabulary.
* Improve recognition accuracy with larger datasets.
* Add more facial-expression and grammatical markers.
* Support continuous sentence-level sign recognition.
* Improve mobile deployment.
* Add more Indian languages.
* Explore real-time two-way communication.
* Develop a lightweight mobile application.

---

## 👩‍💻 Project

**ISL Sign Language Translator**

Built as an AI/ML accessibility solution with a focus on **Indian Sign Language and inclusive communication**.

---

## ⭐ Vision

> **Breaking communication barriers through AI-powered Indian Sign Language translation.**
