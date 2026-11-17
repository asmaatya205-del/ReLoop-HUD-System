"""
ReLoop HUD System
------------------
A real-time ASL (American Sign Language) recognition and translation desktop
application built with PyQt6, OpenCV, MediaPipe and OpenAI.
 
Architecture notes (important for grading / maintenance):
- All camera / video I/O (cv2.VideoCapture objects) is owned and touched
  ONLY by the CameraWorker thread. The GUI thread never opens, releases or
  reads from a VideoCapture object directly. Instead, it queues a small
  "command" via request_*() methods that the worker thread itself executes.
  This avoids race conditions (and the FFmpeg/Qt crashes they caused) when
  two threads touch the same VideoCapture object at the same time.
 
New in this version:
1. Confidence Voting: a small majority-vote smoothing filter over the last
   few model predictions, reducing single-frame noise without adding any
   extra model inference cost (see VOTE_WINDOW).
2. Speech-to-Sign: record a short audio clip, transcribe it with OpenAI
   Whisper, find known ASL words inside the transcribed sentence, and play
   their sign videos back-to-back as a sequence.
3. Learning Mode submenu: Learn (watch & mirror a sign), Quiz (random word,
   no demo video, scored), and Statistics (per-word success rate, persisted
   to a local JSON file).
4. Language selector: a shared dropdown used by BOTH translation features.
   - AI Smart Translate (signs -> sentence -> speech): the recognized
     ASL keywords are translated into a natural sentence IN THE SELECTED
     LANGUAGE before being spoken.
   - Speech-to-Sign (speech -> text -> signs): the spoken audio is assumed
     to be in the selected language; it is transcribed, then translated to
     English internally so it can be matched against the (English) ASL
     vocabulary, while the original heard sentence is still shown to the
     user for reference.
"""
 
import sys
import os
import json
import random
import threading
import time
from datetime import datetime
 
import cv2
import numpy as np
import mediapipe as mp
import openai
import pygame
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
 
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy,
                             QFileDialog, QGridLayout, QProgressBar, QLineEdit,
                             QComboBox,QScrollArea)
from PyQt6.QtGui import QImage, QPixmap, QFont
 
# ---------------------------------------------------------------------------
# Environment / global setup
# ---------------------------------------------------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    # AI Smart Translate and Speech-to-Sign will fail gracefully later if
    # this is missing; we just warn early so it's obvious during development.
    print("WARNING: OPENAI_API_KEY is not set. AI-powered features will not work.")
 
pygame.mixer.init()
 
# Load the trained Keras models used for sign recognition and facial emotion
# recognition. The app cannot function without these, so we exit on failure.
try:
    asl_model = load_model('asl_dual_hands_model.keras')
    emotion_model = load_model('emotion_model.keras')
    print("--- HUD System: Models Loaded Successfully ---")
except Exception as e:
    print(f"FATAL: Error loading models: {e}")
    sys.exit(1)
 
# Vocabulary the ASL model was trained to recognize.
actions = np.array([
    'A', 'All Done', 'E', 'G', 'Hello', 'Help', 'I', 'Improve',
    'L', 'Life', 'M', 'N', 'Name', 'Need', 'No', 'Not', 'Nothing',
    'O', 'Profesor', 'S', 'T', 'Technology', 'Use', 'War', 'Y', 'Yes'
])
emotion_labels = ['Happy', 'Neutral', 'Sad']
 
# Subset of words that have demo/sign videos available under videos/, used
# both for the "Learn" word grid and as the random pool for "Quiz" mode.
LEARNING_WORDS = [
    'A', 'All Done', 'E', 'G', 'Hello', 'Help',
    'I', 'L', 'Life', 'M', 'N', 'Name',
    'No', 'O', 'S', 'T', 'Technology', 'War',
    'Y', 'Yes'
]
 
# Number of consecutive model predictions that must agree before a sign is
# accepted as "confirmed". This is the Confidence Voting / smoothing filter:
# it costs no extra inference time (the predictions are already computed
# every frame), it only adds a small buffering delay before a word is
# accepted into the transcript.
VOTE_WINDOW = 3
 
# Local file used to persist per-word Quiz statistics across sessions.
STATS_FILE = "learning_stats.json"
 
# Languages offered in the shared language dropdown. Each entry is:
#   (display_name_in_combo, whisper_language_code, language_name_for_gpt_prompt)
# - whisper_language_code: passed as the `language` hint to OpenAI Whisper
#   for more accurate transcription of the spoken audio in Speech-to-Sign.
# - language_name_for_gpt_prompt: a plain-English language name used inside
#   GPT prompts (both to ask for an output sentence in that language, and to
#   ask GPT to translate transcribed text *into* English for ASL matching).
LANGUAGES = [
    ("English",   "en", "English"),
    ("العربية",   "ar", "Arabic"),
    ("Français",  "fr", "French"),
    ("Español",   "es", "Spanish"),
    ("Deutsch",   "de", "German"),
    ("Türkçe",    "tr", "Turkish"),
]
DEFAULT_LANGUAGE_INDEX = 0  # English
 
 
# ---------------------------------------------------------------------------
# CameraWorker: background thread that owns the camera / video pipeline
# ---------------------------------------------------------------------------
class CameraWorker(QThread):
    """
    Background thread responsible for:
      - Reading frames from the live webcam, an uploaded video file, or a
        looping demo/sign video (or a sequence of sign videos).
      - Running MediaPipe hand and face landmark detection.
      - Running the ASL recognition model and the emotion recognition model.
      - Smoothing raw predictions with a majority-vote filter.
      - Emitting processed frames and inference results back to the GUI
        thread via Qt signals.
 
    Thread-safety design:
      All cv2.VideoCapture objects (live_cap, video_file_cap, demo_video_cap)
      are created, used and released exclusively inside run(), i.e. inside
      this worker thread. The GUI thread never touches them directly.
 
      To change what the worker is doing, the GUI thread calls one of the
      request_*() methods below. These methods only store a small command
      tuple under a lock; the actual OpenCV work happens later, inside the
      worker thread, in _process_pending_command().
    """
    frame_processed_signal = pyqtSignal(np.ndarray, str)
    inference_signal = pyqtSignal(str, float, str)
    error_signal = pyqtSignal(str)
 
    def __init__(self):
        super().__init__()
        self.running = True
        self.current_mode = "LIVE"
        self.target_training_word = ""
 
        self._lock = threading.Lock()
        self._pending_command = None  # (mode, payload, word_or_None)
 
        # These are created inside run() so they belong to the worker thread.
        self.live_cap = None
        self.video_file_cap = None
        self.demo_video_cap = None  # shared by demo / text-to-sign / sign-sequence playback
        self.hands = None
        self.face_mesh = None
 
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
 
        self.sequence = []
        self.frame_count = 0
        self.current_emotion = "Neutral"
 
        # Confidence voting buffer: list of (word, confidence) tuples for the
        # last VOTE_WINDOW raw predictions.
        self.prediction_buffer = []
 
        # State for playing a sequence of sign-language videos back to back
        # (used by the Speech-to-Sign feature).
        self.sign_sequence = []       # list of (video_path, word)
        self.sign_sequence_index = 0
 
    # -----------------------------------------------------------------
    # Public API: called ONLY from the GUI thread.
    # These never touch OpenCV objects directly - they just queue a command.
    # -----------------------------------------------------------------
    def request_live(self):
        """Switch back to the live webcam feed."""
        with self._lock:
            self._pending_command = ("LIVE", None, None)
 
    def request_video_file(self, path):
        """Open and analyze an uploaded video file."""
        with self._lock:
            self._pending_command = ("VIDEO_FILE", path, None)
 
    def request_demo_video(self, path, word):
        """Play a looping 'watch & learn' demo video for a chosen word."""
        with self._lock:
            self._pending_command = ("DEMO_SHOWING", path, word)
 
    def request_demo_missing(self, word):
        """No demo video file exists for this word - show a placeholder."""
        with self._lock:
            self._pending_command = ("DEMO_MISSING", None, word)
 
    def request_training(self, word):
        """Start live training mode: try to match the user's sign to `word`."""
        with self._lock:
            self._pending_command = ("TRAINING_ACTIVE", None, word)
 
    def request_text_to_sign(self, path, word):
        """Play the sign video for a single word typed by the user."""
        with self._lock:
            self._pending_command = ("TEXT_TO_SIGN", path, word)
 
    def request_sign_sequence(self, sequence):
        """
        Play a list of sign videos back-to-back, in order.
        sequence: list of (video_path, word) tuples.
        Used by the Speech-to-Sign feature to "speak" a whole sentence.
        """
        with self._lock:
            self._pending_command = ("SIGN_SEQUENCE", sequence, None)
 
    # -----------------------------------------------------------------
    # Internal: executed ONLY inside run(), i.e. the worker thread.
    # -----------------------------------------------------------------
    def _process_pending_command(self):
        """Pick up the latest command (if any) and apply it."""
        with self._lock:
            cmd = self._pending_command
            self._pending_command = None
        if cmd is None:
            return
 
        mode, payload, word = cmd
 
        # Reset sign-sequence state unless we are starting a new sequence.
        if mode != "SIGN_SEQUENCE":
            self.sign_sequence = []
            self.sign_sequence_index = 0
 
        # Release whichever capture objects are no longer needed for the
        # new mode. This always happens inside this thread, so there is no
        # race with a concurrent read() call from elsewhere.
        if mode != "VIDEO_FILE" and self.video_file_cap is not None:
            self.video_file_cap.release()
            self.video_file_cap = None
        if mode not in ("DEMO_SHOWING", "TEXT_TO_SIGN") and self.demo_video_cap is not None:
            self.demo_video_cap.release()
            self.demo_video_cap = None
 
        if mode == "LIVE":
            self.current_mode = "LIVE"
 
        elif mode == "VIDEO_FILE":
            cap = cv2.VideoCapture(payload)
            if cap.isOpened():
                self.video_file_cap = cap
                self.sequence = []
                self.prediction_buffer = []
                self.current_mode = "VIDEO_FILE"
            else:
                self.error_signal.emit(f"Could not open video file: {payload}")
                self.current_mode = "LIVE"
 
        elif mode == "DEMO_SHOWING":
            cap = cv2.VideoCapture(payload)
            if cap.isOpened():
                self.demo_video_cap = cap
                self.target_training_word = word
                self.current_mode = "DEMO_SHOWING"
            else:
                self.target_training_word = word
                self.current_mode = "DEMO_MISSING"
 
        elif mode == "DEMO_MISSING":
            self.target_training_word = word
            self.current_mode = "DEMO_MISSING"
 
        elif mode == "TRAINING_ACTIVE":
            self.target_training_word = word
            self.sequence = []
            self.prediction_buffer = []
            self.current_mode = "TRAINING_ACTIVE"
 
        elif mode == "TEXT_TO_SIGN":
            cap = cv2.VideoCapture(payload)
            if cap.isOpened():
                self.demo_video_cap = cap
                self.target_training_word = word
                self.current_mode = "TEXT_TO_SIGN"
            else:
                self.error_signal.emit(f"Could not open sign video for '{word}'")
                self.current_mode = "LIVE"
 
        elif mode == "SIGN_SEQUENCE":
            self.sign_sequence = payload  # list of (path, word) tuples
            self.sign_sequence_index = 0
            self.sequence = []
            self.prediction_buffer = []
            self._load_sequence_clip()
            self.current_mode = "SIGN_SEQUENCE"
 
    def _load_sequence_clip(self):
        """
        Open the video clip for the current index in self.sign_sequence.
        Only ever called from inside this worker thread.
        """
        if self.demo_video_cap is not None:
            self.demo_video_cap.release()
            self.demo_video_cap = None
 
        if self.sign_sequence_index >= len(self.sign_sequence):
            self.current_mode = "LIVE"
            return
 
        clip_path, word = self.sign_sequence[self.sign_sequence_index]
        self.target_training_word = word
        cap = cv2.VideoCapture(clip_path)
        if cap.isOpened():
            self.demo_video_cap = cap
        else:
            self.demo_video_cap = None  # missing clip - will be skipped in run()
 
    def run(self):
        # Camera and MediaPipe objects are created here so they are owned by
        # this thread, not by the GUI thread that constructed CameraWorker().
        try:
            self.live_cap = cv2.VideoCapture(0)
            self.hands = self.mp_hands.Hands(min_detection_confidence=0.7,
                                              min_tracking_confidence=0.5, max_num_hands=2)
            self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)
        except Exception as e:
            self.error_signal.emit(f"Failed to initialize camera/MediaPipe: {e}")
            return
 
        while self.running:
            try:
                # Apply any mode-switch request queued by the GUI thread.
                self._process_pending_command()
 
                self.frame_count += 1
                frame = None
                header_text = "LIVE CAMERA FEED"
 
                if self.current_mode in ("LIVE", "TRAINING_ACTIVE"):
                    ret, frame = self.live_cap.read()
                    if not ret:
                        frame = None
                    if self.current_mode == "TRAINING_ACTIVE":
                        header_text = f"TRAINING ACTIVE: DO THE SIGN FOR '{self.target_training_word.upper()}'"
 
                elif self.current_mode in ("DEMO_SHOWING", "TEXT_TO_SIGN"):
                    # Both modes loop a short reference video; only the
                    # on-screen header text differs.
                    if self.demo_video_cap is not None and self.demo_video_cap.isOpened():
                        ret, frame = self.demo_video_cap.read()
                        if not ret:
                            self.demo_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            time.sleep(0.01)
                            continue
                        if self.current_mode == "DEMO_SHOWING":
                            header_text = f"WATCH & LEARN GESTURE: {self.target_training_word.upper()}"
                        else:
                            header_text = f"SIGN FOR: {self.target_training_word.upper()}"
                        self.frame_processed_signal.emit(frame.copy(), header_text)
                        time.sleep(0.03)
                        continue
                    else:
                        frame = None
 
                elif self.current_mode == "DEMO_MISSING":
                    # No demo asset on disk - show a placeholder frame instead
                    # of freezing on a black screen.
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, f"[ Loop Avatar for '{self.target_training_word}' ]",
                                (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
                    self.frame_processed_signal.emit(blank_frame, "AVATAR ASSET MISSING")
                    time.sleep(0.05)
                    continue
 
                elif self.current_mode == "VIDEO_FILE":
                    if self.video_file_cap is not None and self.video_file_cap.isOpened():
                        ret, frame = self.video_file_cap.read()
                        if not ret:
                            self.video_file_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.sequence = []
                            self.prediction_buffer = []
                            time.sleep(0.01)
                            continue
                        header_text = "VIDEO FILE ANALYSIS"
                    else:
                        frame = None
 
                elif self.current_mode == "SIGN_SEQUENCE":
                    if self.demo_video_cap is not None and self.demo_video_cap.isOpened():
                        ret, frame = self.demo_video_cap.read()
                        if not ret:
                            # Current clip finished - advance to the next word.
                            self.sign_sequence_index += 1
                            self._load_sequence_clip()
                            time.sleep(0.05)
                            continue
                        header_text = (f"SIGN SEQUENCE ({self.sign_sequence_index + 1}/"
                                       f"{len(self.sign_sequence)}): {self.target_training_word.upper()}")
                        self.frame_processed_signal.emit(frame.copy(), header_text)
                        time.sleep(0.03)
                        continue
                    else:
                        # Missing clip for this word in the sequence - skip it.
                        self.sign_sequence_index += 1
                        self._load_sequence_clip()
                        time.sleep(0.3)
                        continue
 
                # Defensive check: skip this iteration if we have no valid frame.
                if frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue
 
                h, w, c = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = self.hands.process(rgb_frame)
                lh_lms, rh_lms = np.zeros(21 * 3), np.zeros(21 * 3)
 
                if hand_results.multi_hand_landmarks:
                    for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        label = hand_results.multi_handedness[i].classification[0].label
                        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                        if label == 'Left':
                            lh_lms = coords
                        else:
                            rh_lms = coords
 
                # Run emotion detection every 15 frames (skip during active
                # training to save CPU for the more important ASL inference).
                if self.current_mode != "TRAINING_ACTIVE" and (self.frame_count % 15 == 0):
                    face_results = self.face_mesh.process(rgb_frame)
                    if face_results.multi_face_landmarks:
                        face_lms = face_results.multi_face_landmarks[0].landmark
                        nose = face_lms[1]
                        face_data = [coord for lm in face_lms
                                     for coord in [lm.x - nose.x, lm.y - nose.y, lm.z - nose.z]]
                        face_input = np.array(face_data).reshape(1, -1) * 100
                        emo_pred = emotion_model.predict(face_input, verbose=0)[0]
                        self.current_emotion = emotion_labels[np.argmax(emo_pred)]
 
                # Maintain a rolling 20-frame window for the sequence model.
                full_data = np.concatenate([lh_lms, rh_lms])
                self.sequence.append(full_data)
                self.sequence = self.sequence[-20:]
 
                if len(self.sequence) == 20:
                    res = asl_model.predict(np.expand_dims(np.array(self.sequence), axis=0), verbose=0)[0]
                    predicted_idx = np.argmax(res)
                    raw_confidence = float(res[predicted_idx])
                    raw_word = actions[predicted_idx]
 
                    # --- Confidence voting / smoothing -------------------
                    # Buffer the last VOTE_WINDOW raw predictions and only
                    # emit a "confirmed" prediction once a majority of them
                    # agree on the same word. This filters out single-frame
                    # noise without any extra model inference - the model
                    # already runs every frame regardless.
                    self.prediction_buffer.append((raw_word, raw_confidence))
                    if len(self.prediction_buffer) > VOTE_WINDOW:
                        self.prediction_buffer.pop(0)
 
                    words_only = [w for w, _ in self.prediction_buffer]
                    majority_word = max(set(words_only), key=words_only.count)
                    agreement = words_only.count(majority_word)
                    majority_needed = VOTE_WINDOW // 2 + 1
 
                    if agreement >= majority_needed:
                        agreeing_confidences = [conf for w, conf in self.prediction_buffer if w == majority_word]
                        avg_confidence = sum(agreeing_confidences) / len(agreeing_confidences)
                        self.inference_signal.emit(majority_word, avg_confidence, self.current_emotion)
 
                self.frame_processed_signal.emit(frame.copy(), header_text)
                time.sleep(0.02)
 
            except Exception as e:
                # A single bad frame must not kill the whole worker thread.
                self.error_signal.emit(f"Error while processing frame: {e}")
                time.sleep(0.1)
 
        # Final cleanup - still inside this thread, the thread that created
        # these capture objects in the first place.
        if self.live_cap is not None:
            self.live_cap.release()
        if self.video_file_cap is not None:
            self.video_file_cap.release()
        if self.demo_video_cap is not None:
            self.demo_video_cap.release()
 
    def stop(self):
        """Signal the worker loop to exit. Actual cleanup happens in run()."""
        self.running = False
 
 
# ---------------------------------------------------------------------------
# AISpeechThread: background thread for OpenAI text + text-to-speech
# (forward direction: recognized signs -> natural sentence -> spoken audio)
# ---------------------------------------------------------------------------
class AISpeechThread(QThread):
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
 
    def __init__(self, word_list, emotion, target_language_name="English"):
        super().__init__()
        self.word_list = word_list
        self.emotion = emotion
        # Plain-English name of the language to translate the final sentence
        # into, e.g. "Arabic", "French". Comes from the shared language
        # dropdown in the GUI.
        self.target_language_name = target_language_name
 
    def run(self):
        if not self.word_list:
            return
        if not openai.api_key:
            self.error_signal.emit("OpenAI API key is not configured (OPENAI_API_KEY).")
            return
 
        raw_text = " ".join(self.word_list)
        try:
            client = openai.OpenAI(api_key=openai.api_key)
            prompt = (f"Translate these ASL keywords into a natural sentence written "
                      f"in {self.target_language_name}. "
                      f"Speaker mood: {self.emotion}. Keywords: {raw_text}")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": ("You are ReLoop AI. Translate ASL keywords into one natural "
                                 f"sentence written in {self.target_language_name}. "
                                 "Output ONLY the sentence, in that language, nothing else.")},
                    {"role": "user", "content": prompt}
                ]
            )
            smart_sentence = response.choices[0].message.content.strip()
 
            pygame.mixer.music.stop()
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass  # unload() may not exist on older pygame versions
 
            # tts-1 auto-detects the spoken language from the input text, so
            # no extra parameter is needed here - it will read the sentence
            # out loud in whatever language smart_sentence ended up in.
            speech_response = client.audio.speech.create(model="tts-1", voice="shimmer", input=smart_sentence)
            filename = f"speech_{int(time.time())}.mp3"
            speech_response.write_to_file(filename)
 
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            self.finished_signal.emit(smart_sentence)
        except Exception as e:
            # IMPORTANT: always emit error_signal on failure so the GUI can
            # re-enable the translate button instead of staying stuck on
            # "COMPILING AI..." forever.
            self.error_signal.emit(str(e))
 
 
# ---------------------------------------------------------------------------
# SpeechToSignThread: background thread for the reverse direction
# (spoken audio -> transcribed text -> matched ASL words)
# ---------------------------------------------------------------------------
class SpeechToSignThread(QThread):
    """
    Records a short clip from the default microphone, sends it to OpenAI
    Whisper for transcription, and emits the resulting text.
 
    If the selected spoken language is not English, the transcribed text is
    additionally translated to English (via a quick GPT call) so it can be
    matched against the English ASL vocabulary. Both the original heard text
    and the English text used for matching are emitted, so the GUI can show
    the user what was actually heard.
 
    Requires the optional dependencies `sounddevice` and `scipy`
    (pip install sounddevice scipy). They are imported lazily inside run()
    so the rest of the application still works even if they are not
    installed - only this specific feature will report an error.
    """
    # (original_heard_text, english_text_for_matching)
    transcribed_signal = pyqtSignal(str, str)
    error_signal = pyqtSignal(str)
 
    RECORD_SECONDS = 4
    SAMPLE_RATE = 16000
 
    def __init__(self, whisper_language_code="en", language_name="English"):
        super().__init__()
        # ISO-639-1 code passed as a hint to Whisper for better accuracy,
        # e.g. "ar", "fr". Comes from the shared language dropdown.
        self.whisper_language_code = whisper_language_code
        # Plain-English name used when asking GPT to translate the
        # transcribed text into English, e.g. "Arabic", "French".
        self.language_name = language_name
 
    def run(self):
        if not openai.api_key:
            self.error_signal.emit("OpenAI API key is not configured (OPENAI_API_KEY).")
            return
 
        wav_path = None
        try:
            import sounddevice as sd
            from scipy.io.wavfile import write as wav_write
        except ImportError:
            self.error_signal.emit(
                "Missing dependency: install 'sounddevice' and 'scipy' "
                "(pip install sounddevice scipy) to use Speech-to-Sign.")
            return
 
        try:
            recording = sd.rec(int(self.RECORD_SECONDS * self.SAMPLE_RATE),
                                samplerate=self.SAMPLE_RATE, channels=1, dtype='int16')
            sd.wait()  # block until the recording finishes
 
            wav_path = f"voice_input_{int(time.time())}.wav"
            wav_write(wav_path, self.SAMPLE_RATE, recording)
 
            client = openai.OpenAI(api_key=openai.api_key)
            with open(wav_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=self.whisper_language_code
                )
            text = transcript.text.strip()
 
            # If the spoken language is already English, no translation step
            # is needed - the text can be matched against the ASL vocabulary
            # directly.
            if self.language_name.lower() == "english" or not text:
                english_text = text
            else:
                english_text = self._translate_to_english(client, text)
 
            self.transcribed_signal.emit(text, english_text)
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            # Clean up the temporary audio file regardless of success/failure.
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
 
    def _translate_to_english(self, client, text):
        """
        Translate transcribed, non-English text into English so it can be
        matched against the (English) ASL vocabulary. Falls back to the
        original text if the translation call fails, so a translation
        hiccup never blocks sign matching entirely.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": ("Translate the user's message into English. "
                                 "Output ONLY the English translation, nothing else.")},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return text
 
 
# ---------------------------------------------------------------------------
# ReLoopHUDApp: main GUI window
# ---------------------------------------------------------------------------
class ReLoopHUDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RELOOP PRO SYSTEM - HUD v3.1 (Voting + Speech-to-Sign + Quiz + Language)")
        self.resize(1200, 780)
 
        self.sentence_words = []
        self.threshold = 0.80
        # Tracked on the GUI side only, so we never need to read state back
        # from the worker thread (which would itself be a minor race risk).
        self.selected_learning_word = ""
 
        # --- Quiz mode state ---
        self.in_quiz_mode = False
        self.current_quiz_word = ""
        self.last_quiz_matched = False
        self.quiz_score_correct = 0
        self.quiz_score_total = 0
        self.quiz_word_pool = LEARNING_WORDS
 
        # --- Statistics state (persisted to STATS_FILE as JSON) ---
        self.learning_stats = self.load_stats()
 
        # Pick a default theme automatically based on the current time of day.
        current_hour = datetime.now().hour
        if 7 <= current_hour <= 18:
            self.is_dark_mode = False
            self.current_theme_file = "light_theme.css"
        else:
            self.is_dark_mode = True
            self.current_theme_file = "dark_theme.css"
 
        self.init_ui()
        self.load_theme(self.current_theme_file)
 
        self.camera_worker = CameraWorker()
        self.camera_worker.frame_processed_signal.connect(self.update_camera_screen)
        self.camera_worker.inference_signal.connect(self.handle_ai_inference)
        self.camera_worker.error_signal.connect(self.handle_worker_error)
        self.camera_worker.start()
 
    # -----------------------------------------------------------------
    # Theme handling
    # -----------------------------------------------------------------
    def load_theme(self, filename):
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
        else:
            print(f"Theme file missing: {filename}")
 
    def toggle_theme(self):
        if self.is_dark_mode:
            self.current_theme_file = "light_theme.css"
            self.btn_theme_toggle.setText("🌙 DARK MODE")
            self.is_dark_mode = False
        else:
            self.current_theme_file = "dark_theme.css"
            self.btn_theme_toggle.setText("☀️ LIGHT MODE")
            self.is_dark_mode = True
 
        self.load_theme(self.current_theme_file)
        self.transcript_val.setStyleSheet("")
 
    # -----------------------------------------------------------------
    # Shared language selector helpers
    # -----------------------------------------------------------------
    def get_selected_language(self):
        """
        Returns (display_name, whisper_code, gpt_language_name) for whatever
        is currently selected in the shared language dropdown. Falls back to
        English if, for any reason, nothing is selected yet.
        """
        idx = self.language_selector.currentIndex()
        if idx < 0 or idx >= len(LANGUAGES):
            return LANGUAGES[DEFAULT_LANGUAGE_INDEX]
        return LANGUAGES[idx]
 
    # -----------------------------------------------------------------
    # Statistics persistence (Quiz mode)
    # -----------------------------------------------------------------
    def load_stats(self):
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Could not read {STATS_FILE}: {e}")
                return {}
        return {}
 
    def save_stats(self):
        try:
            with open(STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.learning_stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Could not save {STATS_FILE}: {e}")
 
    def record_quiz_attempt(self, word, success):
        entry = self.learning_stats.get(word, {"attempts": 0, "successes": 0})
        entry["attempts"] += 1
        if success:
            entry["successes"] += 1
        self.learning_stats[word] = entry
        self.save_stats()
 
    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------
    def init_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("central")
        self.setCentralWidget(central_widget)
 
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
 
        # Main camera / video display.
        self.camera_view = QLabel()
        self.camera_view.setObjectName("camera_view")
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setMinimumSize(640, 480)
        self.camera_view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        main_layout.addWidget(self.camera_view, stretch=2)
 
        # Side control panel with Scroll Area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setFrameShape(scroll_area.frameShape().NoFrame) 
        self.side_panel = QWidget()
        self.side_panel.setObjectName("side_panel")
        self.side_layout = QVBoxLayout(self.side_panel)
        self.side_layout.setContentsMargins(5, 5, 5, 5) 
        self.side_layout.setSpacing(12)
        
        scroll_area.setWidget(self.side_panel)
 
        # Theme toggle button.
        toggle_text = "🌙 DARK MODE" if not self.is_dark_mode else "☀️ LIGHT MODE"
        self.btn_theme_toggle = QPushButton(toggle_text)
        self.btn_theme_toggle.setObjectName("themeToggleBtn")
        self.btn_theme_toggle.clicked.connect(self.toggle_theme)
        self.side_layout.addWidget(self.btn_theme_toggle)
 
        # ============================ Shared language selector ============================
        # Used by BOTH translation features:
        #  - AI Smart Translate: output sentence language.
        #  - Speech-to-Sign: assumed spoken-input language (translated to
        #    English internally for ASL word matching).
        self.language_box = QWidget()
        self.language_box.setProperty("class", "StatusBox")
        lang_layout = QVBoxLayout(self.language_box)
        lang_layout.setContentsMargins(8, 8, 8, 8)
        lang_layout.setSpacing(6)
 
        lang_title = QLabel("🌐 LANGUAGE")
        lang_title.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        lang_layout.addWidget(lang_title)
 
        self.language_selector = QComboBox()
        for display_name, _code, _gpt_name in LANGUAGES:
            self.language_selector.addItem(display_name)
        self.language_selector.setCurrentIndex(DEFAULT_LANGUAGE_INDEX)
        self.language_selector.setFont(QFont("Consolas", 10))
        lang_layout.addWidget(self.language_selector)
 
        self.side_layout.addWidget(self.language_box)
 
        self.title_label = QLabel("📊 INTELLIGENCE DESK")
        self.title_label.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        self.side_layout.addWidget(self.title_label)
 
        # ============================ Translation mode ============================
        self.translation_widgets = QWidget()
        trans_w_layout = QVBoxLayout(self.translation_widgets)
        trans_w_layout.setContentsMargins(0, 0, 0, 0)
        trans_w_layout.setSpacing(10)
 
        self.btn_upload_img = QPushButton("📁 UPLOAD ASL IMAGE")
        self.btn_upload_img.setObjectName("uploadBtn")
        self.btn_upload_img.clicked.connect(self.upload_image)
        trans_w_layout.addWidget(self.btn_upload_img)
 
        self.btn_upload_vid = QPushButton("🎞️ UPLOAD ASL VIDEO")
        self.btn_upload_vid.setObjectName("uploadBtn")
        self.btn_upload_vid.clicked.connect(self.upload_video)
        trans_w_layout.addWidget(self.btn_upload_vid)
 
        self.btn_switch_learn = QPushButton("🎓 ENTER LEARNING MODE")
        self.btn_switch_learn.clicked.connect(self.open_learning_mode)
        trans_w_layout.addWidget(self.btn_switch_learn)
        self.side_layout.addWidget(self.translation_widgets)
 
        # ============================ Learning mode: top menu ============================
        self.learning_menu_widgets = QWidget()
        menu_layout = QVBoxLayout(self.learning_menu_widgets)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.setSpacing(10)
 
        menu_title = QLabel("Choose a mode:")
        menu_title.setFont(QFont("Consolas", 10))
        menu_layout.addWidget(menu_title)
 
        self.btn_menu_learn = QPushButton("📖 LEARN")
        self.btn_menu_learn.clicked.connect(self.open_learn_submode)
        menu_layout.addWidget(self.btn_menu_learn)
 
        self.btn_menu_quiz = QPushButton("📝 QUIZ")
        self.btn_menu_quiz.clicked.connect(self.enter_quiz_mode)
        menu_layout.addWidget(self.btn_menu_quiz)
 
        self.btn_menu_stats = QPushButton("📊 STATISTICS")
        self.btn_menu_stats.clicked.connect(self.show_statistics)
        menu_layout.addWidget(self.btn_menu_stats)
 
        self.btn_exit_learn = QPushButton("🔙 EXIT LEARNING MODE")
        self.btn_exit_learn.clicked.connect(self.exit_learning_mode)
        menu_layout.addWidget(self.btn_exit_learn)
 
        self.learning_menu_widgets.setVisible(False)
        self.side_layout.addWidget(self.learning_menu_widgets)
 
        # ============================ Learning mode: Learn submode ============================
        self.learning_widgets = QWidget()
        learn_w_layout = QVBoxLayout(self.learning_widgets)
        learn_w_layout.setContentsMargins(0, 0, 0, 0)
        learn_w_layout.setSpacing(8)
 
        choose_lbl = QLabel("Select word to learn:")
        choose_lbl.setFont(QFont("Consolas", 10))
        learn_w_layout.addWidget(choose_lbl)
 
        grid_words = QGridLayout()
        for idx, word in enumerate(LEARNING_WORDS):
            btn = QPushButton(word)
            btn.setProperty("class", "WordLearnBtn")
            btn.clicked.connect(lambda checked, w=word: self.select_learning_word(w))
            grid_words.addWidget(btn, idx // 2, idx % 2)
        learn_w_layout.addLayout(grid_words)
 
        self.btn_start_training = QPushButton("🎯 START TRAINING NOW")
        self.btn_start_training.clicked.connect(self.start_user_training)
        self.btn_start_training.setEnabled(False)
        learn_w_layout.addWidget(self.btn_start_training)
 
        self.btn_back_from_learn = QPushButton("🔙 BACK TO MENU")
        self.btn_back_from_learn.clicked.connect(self.back_to_learning_menu)
        learn_w_layout.addWidget(self.btn_back_from_learn)
 
        self.learning_widgets.setVisible(False)
        self.side_layout.addWidget(self.learning_widgets)
 
        # ============================ Learning mode: Quiz submode ============================
        self.quiz_widgets = QWidget()
        quiz_layout = QVBoxLayout(self.quiz_widgets)
        quiz_layout.setContentsMargins(0, 0, 0, 0)
        quiz_layout.setSpacing(8)
 
        self.quiz_word_label = QLabel("Sign this word:\n-")
        self.quiz_word_label.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self.quiz_word_label.setWordWrap(True)
        quiz_layout.addWidget(self.quiz_word_label)
 
        self.quiz_score_label = QLabel("Score: 0/0")
        self.quiz_score_label.setFont(QFont("Consolas", 10))
        quiz_layout.addWidget(self.quiz_score_label)
 
        self.btn_quiz_next = QPushButton("➡️ NEXT WORD")
        self.btn_quiz_next.clicked.connect(self.quiz_next_word)
        quiz_layout.addWidget(self.btn_quiz_next)
 
        self.btn_back_from_quiz = QPushButton("🔙 BACK TO MENU")
        self.btn_back_from_quiz.clicked.connect(self.back_to_learning_menu)
        quiz_layout.addWidget(self.btn_back_from_quiz)
 
        self.quiz_widgets.setVisible(False)
        self.side_layout.addWidget(self.quiz_widgets)
 
        # ============================ Learning mode: Statistics submode ============================
        self.stats_widgets = QWidget()
        stats_layout = QVBoxLayout(self.stats_widgets)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(8)
 
        stats_title = QLabel("Per-word Quiz accuracy:")
        stats_title.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        stats_layout.addWidget(stats_title)
 
        self.stats_text = QLabel("No quiz attempts recorded yet.")
        self.stats_text.setFont(QFont("Consolas", 10))
        self.stats_text.setWordWrap(True)
        stats_layout.addWidget(self.stats_text)
 
        self.btn_back_from_stats = QPushButton("🔙 BACK TO MENU")
        self.btn_back_from_stats.clicked.connect(self.back_to_learning_menu)
        stats_layout.addWidget(self.btn_back_from_stats)
 
        self.stats_widgets.setVisible(False)
        self.side_layout.addWidget(self.stats_widgets)
 
        # Shared "back to live feed" button used by several modes.
        self.btn_live = QPushButton("🎥 BACK TO LIVE FEED")
        self.btn_live.clicked.connect(self.switch_to_live)
        self.btn_live.setVisible(False)
        self.side_layout.addWidget(self.btn_live)
 
        # ============================ Live mood (emotion) box ============================
        self.mood_box = QWidget()
        self.mood_box.setProperty("class", "StatusBox")
        mood_layout = QVBoxLayout(self.mood_box)
        mood_title = QLabel("SYSTEM LIVE MOOD:")
        self.mood_val = QLabel("Neutral")
        self.mood_val.setObjectName("mood_val")
        self.mood_val.setFont(QFont("Consolas", 20, QFont.Weight.Bold))
        mood_layout.addWidget(mood_title)
        mood_layout.addWidget(self.mood_val)
        self.side_layout.addWidget(self.mood_box)
 
        # ============================ Training/Quiz accuracy progress bar ============================
        self.progress_box = QWidget()
        self.progress_box.setProperty("class", "StatusBox")
        prog_layout = QVBoxLayout(self.progress_box)
        self.prog_title = QLabel("🎯 MATCH ACCURACY:")
        self.accuracy_bar = QProgressBar()
        self.accuracy_bar.setRange(0, 100)
        self.accuracy_bar.setValue(0)
        prog_layout.addWidget(self.prog_title)
        prog_layout.addWidget(self.accuracy_bar)
        self.progress_box.setVisible(False)
        self.side_layout.addWidget(self.progress_box)
 
        # ============================ Live transcript box ============================
        self.transcript_box = QWidget()
        self.transcript_box.setProperty("class", "StatusBox")
        trans_layout = QVBoxLayout(self.transcript_box)
        self.trans_title = QLabel("📝 LIVE TRANSCRIPT:")
        self.transcript_val = QLabel("Waiting for signs...")
        self.transcript_val.setObjectName("transcript_val")
        self.transcript_val.setWordWrap(True)
        self.transcript_val.setFont(QFont("Consolas", 12))
        trans_layout.addWidget(self.trans_title)
        trans_layout.addWidget(self.transcript_val)
        self.side_layout.addWidget(self.transcript_box)
 
        # ============================ Translation action buttons ============================
        self.action_buttons = QWidget()
        act_layout = QVBoxLayout(self.action_buttons)
        act_layout.setContentsMargins(0, 0, 0, 0)
        act_layout.setSpacing(10)
 
        self.btn_translate = QPushButton("✨ AI SMART TRANSLATE")
        self.btn_translate.setObjectName("translateBtn")
        self.btn_translate.clicked.connect(self.trigger_ai_translation)
        act_layout.addWidget(self.btn_translate)
 
        # Row with "delete last word" + "clear all" buttons.
        clear_row = QHBoxLayout()
 
        self.btn_delete_last = QPushButton("⌫ DELETE LAST")
        self.btn_delete_last.clicked.connect(self.delete_last_word)
        clear_row.addWidget(self.btn_delete_last)
 
        self.btn_clear = QPushButton("🗑️ CLEAR ALL")
        self.btn_clear.clicked.connect(self.clear_transcript)
        clear_row.addWidget(self.btn_clear)
 
        act_layout.addLayout(clear_row)
        self.side_layout.addWidget(self.action_buttons)
 
        # ============================ Reverse translation: Text -> Sign ============================
        self.reverse_box = QWidget()
        self.reverse_box.setProperty("class", "StatusBox")
        rev_layout = QVBoxLayout(self.reverse_box)
        rev_layout.setContentsMargins(8, 8, 8, 8)
        rev_layout.setSpacing(8)
 
        rev_title = QLabel("🔤 TEXT → SIGN")
        rev_title.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        rev_layout.addWidget(rev_title)
 
        rev_input_row = QHBoxLayout()
        self.reverse_input = QLineEdit()
        self.reverse_input.setPlaceholderText("Type a word (e.g. Hello)...")
        self.reverse_input.setFont(QFont("Consolas", 10))
        self.reverse_input.returnPressed.connect(self.show_sign_for_word)
        rev_input_row.addWidget(self.reverse_input)
 
        self.btn_reverse = QPushButton("▶")
        self.btn_reverse.setFixedWidth(36)
        self.btn_reverse.setToolTip("Show sign for typed word")
        self.btn_reverse.clicked.connect(self.show_sign_for_word)
        rev_input_row.addWidget(self.btn_reverse)
        rev_layout.addLayout(rev_input_row)
 
        self.reverse_status = QLabel("")
        self.reverse_status.setFont(QFont("Consolas", 10))
        self.reverse_status.setWordWrap(True)
        rev_layout.addWidget(self.reverse_status)
 
        self.side_layout.addWidget(self.reverse_box)
 
        # ============================ Speech -> Sign (sentence) panel ============================
        self.voice_box = QWidget()
        self.voice_box.setProperty("class", "StatusBox")
        voice_layout = QVBoxLayout(self.voice_box)
        voice_layout.setContentsMargins(8, 8, 8, 8)
        voice_layout.setSpacing(8)
 
        voice_title = QLabel("🎙️ SPEECH → SIGN")
        voice_title.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        voice_layout.addWidget(voice_title)
 
        self.btn_speech = QPushButton(f"🎙️ RECORD ({SpeechToSignThread.RECORD_SECONDS}s)")
        self.btn_speech.clicked.connect(self.trigger_speech_to_sign)
        voice_layout.addWidget(self.btn_speech)
 
        self.voice_status = QLabel("")
        self.voice_status.setFont(QFont("Consolas", 10))
        self.voice_status.setWordWrap(True)
        voice_layout.addWidget(self.voice_status)
 
        self.side_layout.addWidget(self.voice_box)
        # ============================ End of side panel sections ============================
 
        self.side_layout.addStretch()
        main_layout.addWidget(scroll_area, stretch=1) 
    # -----------------------------------------------------------------
    # Slots that receive data emitted by CameraWorker (run on the GUI thread)
    # -----------------------------------------------------------------
    def update_camera_screen(self, frame, header_text):
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(rgb_frame, (10, 10), (550, 35), (0, 0, 0), -1)
 
        text_color = (154, 255, 222) if self.is_dark_mode else (255, 255, 255)
        cv2.putText(rgb_frame, header_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
 
        # .copy() is important: QImage does not copy the underlying buffer
        # by default, and rgb_frame is a local numpy array that could be
        # garbage-collected before Qt is done painting it.
        q_img = QImage(rgb_frame.data, w, h, c * w, QImage.Format.Format_RGB888).copy()
        self.camera_view.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                self.camera_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )
 
    def handle_ai_inference(self, predicted_word, confidence, emotion):
        self.mood_val.setText(emotion)
 
        if self.camera_worker.current_mode == "TRAINING_ACTIVE":
            if predicted_word.lower() == self.camera_worker.target_training_word.lower():
                percentage = int(confidence * 100)
                self.accuracy_bar.setValue(percentage)
                self.transcript_val.setText(
                    f"🎯 Great! You matched '{self.camera_worker.target_training_word}' at {percentage}% accuracy.")
                self.transcript_val.setStyleSheet(
                    "color: #27ae60; font-weight: bold;" if not self.is_dark_mode else "color: #00ff00; font-weight: bold;")
                if self.in_quiz_mode:
                    self.last_quiz_matched = True
            else:
                if predicted_word != 'Nothing' and confidence > 0.30:
                    self.accuracy_bar.setValue(max(5, int(confidence * 12)))
                    self.transcript_val.setText(f"❌ Detected '{predicted_word}' instead. Match your hand!")
                    self.transcript_val.setStyleSheet(
                        "color: #c0392b;" if not self.is_dark_mode else "color: #ff0055;")
                else:
                    self.accuracy_bar.setValue(0)
                    self.transcript_val.setText(
                        f"Waiting for accurate '{self.camera_worker.target_training_word}' sign...")
                    self.transcript_val.setStyleSheet(
                        "color: #d35400;" if not self.is_dark_mode else "color: #ffff00;")
        else:
            if confidence > self.threshold and predicted_word != 'Nothing':
                if len(self.sentence_words) == 0 or predicted_word != self.sentence_words[-1]:
                    self.sentence_words.append(predicted_word)
                    self.transcript_val.setText(" ".join(self.sentence_words))
 
    def handle_worker_error(self, message):
        """Surface worker-thread errors in the UI instead of failing silently."""
        print(f"[CameraWorker Error] {message}")
        self.transcript_val.setText(f"⚠️ {message}")
        self.transcript_val.setStyleSheet("color: #e67e22;")
 
    # -----------------------------------------------------------------
    # Live / translation mode helpers
    # -----------------------------------------------------------------
    def upload_image(self):
        # Static image analysis: just display the chosen frame once. This
        # does not touch any VideoCapture object, so it's safe to run
        # directly on the GUI thread.
        file_path, _ = QFileDialog.getOpenFileName(self, "Open ASL Image", "",
                                                     "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            static_frame = cv2.imread(file_path)
            if static_frame is not None:
                self.btn_live.setVisible(True)
                self.update_camera_screen(static_frame, "STATIC IMAGE ANALYSIS")
            else:
                self.handle_worker_error(f"Could not read image: {file_path}")
 
    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open ASL Video File", "",
                                                     "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.camera_worker.request_video_file(file_path)
            self.btn_live.setVisible(True)
 
    def switch_to_live(self):
        self.camera_worker.request_live()
        self.btn_live.setVisible(False)
        self.btn_start_training.setEnabled(False)
        self.btn_start_training.setText("🎯 START TRAINING NOW")
        self.selected_learning_word = ""
 
        # Reset the reverse translation panel when returning to live mode.
        self.reverse_input.clear()
        self.reverse_status.setText("")
        self.reverse_status.setStyleSheet("")
        self.voice_status.setText("")
        self.voice_status.setStyleSheet("")
 
        in_learning_section = (self.learning_menu_widgets.isVisible() or
                                self.learning_widgets.isVisible() or
                                self.quiz_widgets.isVisible() or
                                self.stats_widgets.isVisible())
        if in_learning_section:
            self.back_to_learning_menu()
        else:
            self.clear_transcript()
 
    # -----------------------------------------------------------------
    # Learning mode: top-level menu navigation
    # -----------------------------------------------------------------
    def open_learning_mode(self):
        self.camera_worker.request_live()
        self.translation_widgets.setVisible(False)
        self.action_buttons.setVisible(False)
        self.reverse_box.setVisible(False)
        self.voice_box.setVisible(False)
        self.btn_live.setVisible(False)
        self.mood_box.setVisible(False)
        self.progress_box.setVisible(False)
        self.title_label.setText("🎓 ACADEMY LEARNING MODE")
        self.trans_title.setText("📊 STATUS:")
        self.transcript_val.setText("Choose Learn, Quiz, or Statistics.")
        self.transcript_val.setStyleSheet("")
 
        self.learning_widgets.setVisible(False)
        self.quiz_widgets.setVisible(False)
        self.stats_widgets.setVisible(False)
        self.learning_menu_widgets.setVisible(True)
 
    def back_to_learning_menu(self):
        """Return to the Learn/Quiz/Statistics chooser from any submode."""
        self.camera_worker.request_live()
        self.in_quiz_mode = False
        self.learning_widgets.setVisible(False)
        self.quiz_widgets.setVisible(False)
        self.stats_widgets.setVisible(False)
        self.learning_menu_widgets.setVisible(True)
        self.mood_box.setVisible(False)
        self.progress_box.setVisible(False)
        self.title_label.setText("🎓 ACADEMY LEARNING MODE")
        self.trans_title.setText("📊 STATUS:")
        self.transcript_val.setText("Choose Learn, Quiz, or Statistics.")
        self.transcript_val.setStyleSheet("")
        self.btn_live.setVisible(False)
 
    def exit_learning_mode(self):
        """Leave the learning section entirely and return to Translate mode."""
        self.camera_worker.request_live()
        self.in_quiz_mode = False
        self.translation_widgets.setVisible(True)
        self.action_buttons.setVisible(True)
        self.reverse_box.setVisible(True)
        self.voice_box.setVisible(True)
        self.learning_menu_widgets.setVisible(False)
        self.learning_widgets.setVisible(False)
        self.quiz_widgets.setVisible(False)
        self.stats_widgets.setVisible(False)
        self.mood_box.setVisible(True)
        self.progress_box.setVisible(False)
        self.title_label.setText("📊 INTELLIGENCE DESK")
        self.trans_title.setText("📝 LIVE TRANSCRIPT:")
        self.transcript_val.setStyleSheet("")
        self.btn_live.setVisible(False)
 
    # -----------------------------------------------------------------
    # Learning mode: "Learn" submode (watch a demo, then mirror it)
    # -----------------------------------------------------------------
    def open_learn_submode(self):
        self.learning_menu_widgets.setVisible(False)
        self.learning_widgets.setVisible(True)
        self.btn_live.setVisible(True)
        self.title_label.setText("📖 LEARN MODE")
        self.transcript_val.setText("Select a word to start loop learning animation.")
        self.transcript_val.setStyleSheet("")
        self.progress_box.setVisible(False)
        self.accuracy_bar.setValue(0)
 
    def select_learning_word(self, word):
        self.selected_learning_word = word
        self.btn_start_training.setEnabled(True)
        self.btn_start_training.setText(f"🎯 START TRAINING FOR '{word.upper()}'")
        self.transcript_val.setText(
            f"Looping avatar animation for '{word}'. Watch and mirror the gesture, then press Start!")
        self.progress_box.setVisible(False)
        self.accuracy_bar.setValue(0)
 
        # Checking file existence in the GUI thread is safe - it's a plain
        # filesystem check, not an OpenCV object shared with the worker.
        video_path = f"videos/{word.lower().replace(' ', '_')}.mp4"
        if os.path.exists(video_path):
            self.camera_worker.request_demo_video(video_path, word)
        else:
            self.camera_worker.request_demo_missing(word)
 
    def start_user_training(self):
        if not self.selected_learning_word:
            return
        self.camera_worker.request_training(self.selected_learning_word)
        self.progress_box.setVisible(True)
        self.accuracy_bar.setValue(0)
        self.transcript_val.setText(
            f"Camera Active! Mirror the avatar sign for '{self.selected_learning_word.upper()}'...")
        self.transcript_val.setStyleSheet(
            "color: #d35400;" if not self.is_dark_mode else "color: #ffff00;")
 
    # -----------------------------------------------------------------
    # Learning mode: "Quiz" submode (random word, no demo, scored)
    # -----------------------------------------------------------------
    def enter_quiz_mode(self):
        self.learning_menu_widgets.setVisible(False)
        self.quiz_widgets.setVisible(True)
        self.btn_live.setVisible(True)
        self.mood_box.setVisible(False)
        self.progress_box.setVisible(True)
        self.title_label.setText("📝 QUIZ MODE")
        self.trans_title.setText("📊 TRAINING STATUS:")
 
        self.in_quiz_mode = True
        self.quiz_score_correct = 0
        self.quiz_score_total = 0
        self.update_quiz_score_label()
        self.pick_new_quiz_word()
 
    def pick_new_quiz_word(self):
        self.current_quiz_word = random.choice(self.quiz_word_pool)
        self.last_quiz_matched = False
        self.quiz_word_label.setText(f"Sign this word:\n{self.current_quiz_word.upper()}")
        self.accuracy_bar.setValue(0)
        self.transcript_val.setText(
            f"No demo this time - try to sign '{self.current_quiz_word.upper()}' from memory!")
        self.transcript_val.setStyleSheet("")
        # Quiz mode reuses the existing TRAINING_ACTIVE pipeline, just
        # without ever showing the user a demo video first.
        self.camera_worker.request_training(self.current_quiz_word)
 
    def quiz_next_word(self):
        if self.current_quiz_word:
            success = self.last_quiz_matched
            self.record_quiz_attempt(self.current_quiz_word, success)
            self.quiz_score_total += 1
            if success:
                self.quiz_score_correct += 1
            self.update_quiz_score_label()
        self.pick_new_quiz_word()
 
    def update_quiz_score_label(self):
        self.quiz_score_label.setText(f"Score: {self.quiz_score_correct}/{self.quiz_score_total}")
 
    # -----------------------------------------------------------------
    # Learning mode: "Statistics" submode
    # -----------------------------------------------------------------
    def show_statistics(self):
        self.learning_menu_widgets.setVisible(False)
        self.stats_widgets.setVisible(True)
        self.title_label.setText("📊 STATISTICS")
        self.refresh_statistics_view()
 
    def refresh_statistics_view(self):
        if not self.learning_stats:
            self.stats_text.setText("No quiz attempts recorded yet.\nTry the Quiz mode first!")
            return
        lines = []
        for word in sorted(self.learning_stats.keys()):
            data = self.learning_stats[word]
            attempts = data.get("attempts", 0)
            successes = data.get("successes", 0)
            rate = (successes / attempts * 100) if attempts else 0
            lines.append(f"{word:<15}{successes}/{attempts}  ({rate:.0f}%)")
        self.stats_text.setText("\n".join(lines))
 
    # -----------------------------------------------------------------
    # AI Smart Translate (forward: signs -> sentence -> speech)
    # -----------------------------------------------------------------
    def trigger_ai_translation(self):
        if not self.sentence_words:
            return
        self.btn_translate.setText("⚡ COMPILING AI...")
        self.btn_translate.setEnabled(False)
 
        _display_name, _whisper_code, gpt_language_name = self.get_selected_language()
        self.ai_thread = AISpeechThread(self.sentence_words, self.camera_worker.current_emotion,
                                         target_language_name=gpt_language_name)
        self.ai_thread.finished_signal.connect(self.on_ai_translation_complete)
        self.ai_thread.error_signal.connect(self.on_ai_translation_error)
        self.ai_thread.start()
 
    def on_ai_translation_complete(self, final_sentence):
        self.sentence_words = [final_sentence]
        self.transcript_val.setText(final_sentence)
        self.transcript_val.setStyleSheet("")
        self.btn_translate.setText("✨ AI SMART TRANSLATE")
        self.btn_translate.setEnabled(True)
 
    def on_ai_translation_error(self, error_message):
        # Always re-enable the button on failure, otherwise it stays stuck
        # on "COMPILING AI..." forever.
        self.transcript_val.setText(f"⚠️ AI Translation failed: {error_message}")
        self.transcript_val.setStyleSheet("color: #e74c3c;")
        self.btn_translate.setText("✨ AI SMART TRANSLATE")
        self.btn_translate.setEnabled(True)
 
    # -----------------------------------------------------------------
    # Transcript editing
    # -----------------------------------------------------------------
    def delete_last_word(self):
        """Remove the most recently recognized word from the transcript."""
        if self.sentence_words:
            self.sentence_words.pop()
            self.transcript_val.setText(
                " ".join(self.sentence_words) if self.sentence_words else "Waiting for signs...")
            self.transcript_val.setStyleSheet("")
 
    def clear_transcript(self):
        self.sentence_words = []
        self.transcript_val.setText("Waiting for signs...")
        self.transcript_val.setStyleSheet("")
 
    # -----------------------------------------------------------------
    # Reverse translation: Text -> Sign (single word)
    # -----------------------------------------------------------------
    def show_sign_for_word(self):
        """
        Look up the word typed by the user in the known vocabulary and, if a
        matching sign video exists, ask the worker thread to play it.
 
        Only a request is queued here (request_text_to_sign) - the actual
        cv2.VideoCapture is opened later, inside the worker thread, to avoid
        the same race condition that previously caused crashes.
        """
        word = self.reverse_input.text().strip()
        if not word:
            return
 
        # Case-insensitive lookup against the known vocabulary.
        matched_word = None
        for action in actions:
            if action.lower() == word.lower():
                matched_word = action
                break
 
        if matched_word is None:
            self.reverse_status.setText(f"❌ '{word}' not in dictionary yet.")
            self.reverse_status.setStyleSheet("color: #ff4444;")
            return
 
        video_path = f"videos/{matched_word.lower().replace(' ', '_')}.mp4"
 
        if not os.path.exists(video_path):
            self.reverse_status.setText(f"⚠️ Video for '{matched_word}' is missing.\nAdd it to videos/ folder.")
            self.reverse_status.setStyleSheet("color: #ffaa00;")
            return
 
        self.camera_worker.request_text_to_sign(video_path, matched_word)
        self.btn_live.setVisible(True)
        self.reverse_status.setText(f"✅ Showing sign for '{matched_word}'")
        self.reverse_status.setStyleSheet("color: #00ff88;")
 
    # -----------------------------------------------------------------
    # Reverse translation: Speech -> Sign (full sentence, via Whisper)
    # -----------------------------------------------------------------
    def trigger_speech_to_sign(self):
        self.btn_speech.setText("🎙️ LISTENING...")
        self.btn_speech.setEnabled(False)
        self.voice_status.setText(f"Recording for {SpeechToSignThread.RECORD_SECONDS} seconds...")
        self.voice_status.setStyleSheet("color: #3498db;")
 
        _display_name, whisper_code, gpt_language_name = self.get_selected_language()
        self.speech_thread = SpeechToSignThread(whisper_language_code=whisper_code,
                                                 language_name=gpt_language_name)
        self.speech_thread.transcribed_signal.connect(self.on_speech_transcribed)
        self.speech_thread.error_signal.connect(self.on_speech_error)
        self.speech_thread.start()
 
    def on_speech_transcribed(self, heard_text, english_text):
        self.btn_speech.setText(f"🎙️ RECORD ({SpeechToSignThread.RECORD_SECONDS}s)")
        self.btn_speech.setEnabled(True)
 
        if not heard_text:
            self.voice_status.setText("⚠️ Could not understand the audio. Try again.")
            self.voice_status.setStyleSheet("color: #ffaa00;")
            return
 
        # Word matching always runs against the English text (either the
        # original heard text if it was already English, or the GPT
        # translation of it) since the ASL vocabulary itself is in English.
        matched_words = self._find_known_words_in_text(english_text)
 
        # Show the user what was actually heard, plus the English version
        # used for matching when a translation step happened.
        heard_line = f"📝 Heard: \"{heard_text}\""
        if english_text and english_text.strip().lower() != heard_text.strip().lower():
            heard_line += f"\n🌐 English: \"{english_text}\""
 
        if not matched_words:
            self.voice_status.setText(f"{heard_line}\n❌ No known signs found in this sentence.")
            self.voice_status.setStyleSheet("color: #ff4444;")
            return
 
        sequence = []
        missing = []
        for word in matched_words:
            video_path = f"videos/{word.lower().replace(' ', '_')}.mp4"
            if os.path.exists(video_path):
                sequence.append((video_path, word))
            else:
                missing.append(word)
 
        if not sequence:
            self.voice_status.setText(f"{heard_line}\n⚠️ Matched words have no sign videos available.")
            self.voice_status.setStyleSheet("color: #ffaa00;")
            return
 
        self.camera_worker.request_sign_sequence(sequence)
        self.btn_live.setVisible(True)
 
        status_msg = f"{heard_line}\n✅ Showing signs for: {' → '.join(matched_words)}"
        if missing:
            status_msg += f"\n⚠️ Missing videos for: {', '.join(missing)}"
        self.voice_status.setText(status_msg)
        self.voice_status.setStyleSheet("color: #00ff88;")
 
    def on_speech_error(self, error_message):
        self.btn_speech.setText(f"🎙️ RECORD ({SpeechToSignThread.RECORD_SECONDS}s)")
        self.btn_speech.setEnabled(True)
        self.voice_status.setText(f"⚠️ {error_message}")
        self.voice_status.setStyleSheet("color: #e74c3c;")
 
    @staticmethod
    def _find_known_words_in_text(text):
        """
        Scan a free-form transcribed sentence for known vocabulary words, in
        the order they appear. Multi-word entries (e.g. "All Done") are
        checked first so they are matched as a whole instead of matching
        just "All".
        """
        text_lower = text.lower()
        sorted_actions = sorted(actions, key=lambda a: -len(a))
        found = []        # (start_index, word)
        used_spans = []   # already-claimed character ranges, to avoid overlap
 
        for action in sorted_actions:
            needle = action.lower()
            start = 0
            while True:
                idx = text_lower.find(needle, start)
                if idx == -1:
                    break
                end = idx + len(needle)
                overlaps = any(not (end <= s or idx >= e) for s, e in used_spans)
                if not overlaps:
                    found.append((idx, action))
                    used_spans.append((idx, end))
                start = end
 
        found.sort(key=lambda item: item[0])
        return [word for _, word in found]
 
    # -----------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------
    def closeEvent(self, event):
        self.camera_worker.stop()
        self.camera_worker.wait()
        event.accept()
 
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ReLoopHUDApp()
    window.show()
    sys.exit(app.exec())
 