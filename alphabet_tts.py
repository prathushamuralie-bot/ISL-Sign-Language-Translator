"""
Sign Language App - Alphabet Detection to Full Word Speech
Buffers detected letters and speaks the FULL name only once,
instead of speaking each letter as it's detected.
"""

import pyttsx3
import time

engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Buffer to hold letters as they get detected from hand signs
letter_buffer = []
last_detection_time = 0
PAUSE_THRESHOLD = 2.0  # seconds of no new letter = word finished


def on_letter_detected(letter):
    """
    Call this EVERY TIME your hand-detection/alphabet model
    recognizes a letter sign. Do NOT call TTS here.
    """
    global last_detection_time
    letter_buffer.append(letter)
    last_detection_time = time.time()
    print("Detected so far:", ''.join(letter_buffer))  # just for display


def check_word_complete():
    """
    Call this in your main loop (every frame).
    If no new letter has come in for PAUSE_THRESHOLD seconds,
    treat the buffer as a finished word and speak it ONCE.
    """
    global letter_buffer, last_detection_time

    if letter_buffer and (time.time() - last_detection_time) > PAUSE_THRESHOLD:
        speak_full_word(letter_buffer)
        letter_buffer = []   # reset for next word


def speak_full_word(letters):
    word = ''.join(letters)
    print("Speaking full word:", word)
    engine.say(word)
    engine.runAndWait()


# ---------------------------------------------------------
# Example: how to plug this into your existing camera loop
# ---------------------------------------------------------
def main_loop_example():
    """
    Replace this simulation with your actual MediaPipe + model
    prediction loop. The key change: call on_letter_detected()
    when a letter is confirmed, and call check_word_complete()
    every frame — NOT engine.say() inside detection.
    """
    simulated_signs = ['K', 'A', 'V', 'I']  # pretend these come from your model

    for letter in simulated_signs:
        on_letter_detected(letter)
        time.sleep(0.5)   # simulate time between signs

    # simulate user pausing after finishing the word
    time.sleep(2.5)
    check_word_complete()


if __name__ == "__main__":
    main_loop_example()