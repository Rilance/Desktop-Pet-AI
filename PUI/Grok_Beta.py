import os
import sys
import threading
import datetime
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QFont, QIcon
import openai
import requests
from pydub import AudioSegment
from pydub.playback import play
import time
from faster_whisper import WhisperModel  # Import faster-whisper
from src.gui import main
import threading
import datetime  # Newly added import statement
import importlib
import sounddevice as sd
import numpy as np
import tempfile
from scipy.io.wavfile import write
import noisereduce as nr

# Ensure temp directory exists
temp_dir = os.path.join(os.getcwd(), "temp")
os.makedirs(temp_dir, exist_ok=True)

config_path = r"E:\\AI\\PUI\\config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)

OPENAI_API_KEY = config.OPENAI_API_KEY

class ApiCallThread(QThread):
    result = Signal(str)

    def __init__(self, input_text, parent=None):
        super(ApiCallThread, self).__init__(parent)
        self.input_text = input_text

    def run(self):
        result_text = call_api(self.input_text)
        self.result.emit(result_text)

def call_stt_api(file_path):
    """
    Send audio file to Faster Whisper API for transcription.
    
    Args:
        file_path (str): Path to the audio file to be transcribed.
    
    Returns:
        str: The transcribed text.
    """
    url = "http://127.0.0.1:9870/transcribe"
    files = {"file": open(file_path, "rb")}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json().get("text", "")
    except requests.RequestException as e:
        print(f"Error during STT API call: {e}")
        return ""

def call_api(input_text, retries=3, timeout=30):
    XAI_API_KEY = OPENAI_API_KEY
    client = openai.OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )
    for _ in range(retries):
        try:
            completion = client.chat.completions.create(
                model="grok-beta",
                messages=[
                    {"role": "system", "content": "You are Firefly, A girl who is steadfast, brave, gentle, kind, intelligent, and witty, always thinking of others."},
                    {"role": "user", "content": input_text},
                ],
                timeout=timeout
            )
            return completion.choices[0].message.content
        except openai.error.Timeout:
            continue
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
    return "API request timed out or an error occurred, please try again later."

def call_tts_api(text, language):
    url = f"http://127.0.0.1:9880?text={text}&text_language={language}"
    response = requests.get(url)
    if 'Content-Type' in response.headers and response.headers['Content-Type'] == 'audio/wav':
        with open("response.wav", "wb") as f:
            f.write(response.content)
        audio = AudioSegment.from_file("response.wav", format="wav")
        play(audio)
    else:
        print("TTS API did not return a valid WAV file")

def detect_language(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return 'zh'
        elif '\u3040' <= char <= '\u309f':
            return 'ja'
        elif '\u30a0' <= char <= '\u30ff':
            return 'ja'
        elif '\uac00' <= char <= '\ud7af':
            return 'ko'
    return 'en'

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.is_listening = False
        self.stt_model = WhisperModel("./faster-whisper-large-v3")
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Send message to FireFly")
        self.setFixedSize(820, 400)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        self.input_textbox = QTextEdit(self)
        self.input_textbox.setPlaceholderText("Enter text")
        layout.addWidget(self.input_textbox)

        self.submit_button = QPushButton("Submit", self)
        self.submit_button.clicked.connect(self.on_submit)
        layout.addWidget(self.submit_button)

        self.stt_button = QPushButton("\ud83c\udfa4 Start Listening", self)
        self.stt_button.clicked.connect(self.toggle_listening)
        layout.addWidget(self.stt_button)

        self.setLayout(layout)

        screen_geometry = QApplication.primaryScreen().availableGeometry()
        screen_center = screen_geometry.center()
        self.move(
            screen_center.x() - self.frameGeometry().width() // 2,
            screen_geometry.bottom() - self.height() - 30
        )

    def toggle_listening(self):
        if self.is_listening:
            self.is_listening = False
            self.stt_button.setText("\ud83c\udfa4 Start Listening")
        else:
            self.is_listening = True
            self.stt_button.setText("\ud83d\uded1 Stop Listening")
            threading.Thread(target=self.start_stt).start()

    def start_stt(self):
        while self.is_listening:
            try:
                # Record audio
                duration = 5  # seconds
                samplerate = 16000  # Whisper requires 16kHz
                audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
                sd.wait()

                # Perform noise reduction
                reduced_noise = nr.reduce_noise(y=audio_data.flatten(), sr=samplerate, n_fft=1024, hop_length=512)
                audio_data = reduced_noise.reshape(audio_data.shape)

                # Save the denoised audio to temp directory
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_filename = os.path.join(temp_dir, f"recording_{timestamp}_denoised.wav")
                write(temp_filename, samplerate, audio_data.astype(np.int16))

                # Transcribe audio
                transcription = call_stt_api(temp_filename)
                if transcription:
                    self.input_textbox.append(f"[STT]: {transcription}")
                    QApplication.processEvents()
    

            except Exception as e:
                print(f"Error during STT: {e}")

    def on_submit(self):
        input_text = self.input_textbox.toPlainText().strip()
        if input_text:
            self.api_thread = ApiCallThread(input_text)
            self.api_thread.result.connect(self.handle_response)
            self.api_thread.start()
            self.input_textbox.clear()

    def handle_response(self, response_text):
        bubble_window.update_bubble(response_text)
        self.update_bubble_position()

    def update_bubble_position(self):
        character_widget = next((w for w in self.findChildren(QLabel) if isinstance(w, QLabel)), None)
        if character_widget:
            character_pos = character_widget.pos()
            bubble_window.move(character_pos.x(), character_pos.y())

class BubbleWindow(QWidget):
    def __init__(self, x, y):
        super().__init__()
        self.initUI(x, y)

    def initUI(self, x, y):
        self.setWindowTitle("Answer")
        self.setGeometry(x, y, 300, 100)
        layout = QVBoxLayout()
        self.bubble_label = QLabel(self)
        self.bubble_label.setAlignment(Qt.AlignLeft)
        font = QFont()
        self.bubble_label.setFont(font)
        self.bubble_label.setWordWrap(True)
        self.bubble_label.setStyleSheet(
            "QLabel { background-color : white; color : black; border: 1px solid black; "
            "border-radius: 10px; padding: 10px; }"
        )
        layout.addWidget(self.bubble_label)
        self.setLayout(layout)
        self.hide()

    def update_bubble(self, response_text):
        if response_text.strip():
            self.bubble_label.setText("")
            self.show()

            def tts_thread(text, language):
                call_tts_api(text, language)

            language = detect_language(response_text)
            char_delay = 0.05 if language == 'en' else 0.08

            for char in response_text:
                self.bubble_label.setText(self.bubble_label.text() + char)
                QApplication.processEvents()
                time.sleep(char_delay)

            threading.Thread(target=tts_thread, args=(response_text, language)).start()
        else:
            self.hide()

def get_time_of_day():
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        return 'morning'
    elif 12 <= current_hour < 17:
        return 'afternoon'
    elif 17 <= current_hour < 21:
        return 'evening'
    else:
        return 'night'

if __name__ == '__main__':
    # Initialize GUI
    main.logger.add("log\\latest.log", rotation="500 MB")
    app = QApplication(sys.argv)
    window = main.MainWindow(app)
    window.show()

    # Create and display the main window
    text_gen_window = MainWindow()
    text_gen_window.show()

    # Create and display the answer bubble window, initial position can be customized (e.g., x=300, y=200)
    bubble_window = BubbleWindow(300, 200)
    bubble_window.show()

    # Use a timer to periodically update the position of the answer bubble box
    timer = QTimer()
    timer.timeout.connect(text_gen_window.update_bubble_position)
    timer.start(1000)  # Update position every second
    
    # Ensure voicePackData contains all necessary keys
    voicePackData = {
        'morning': 'Good morning',
        'afternoon': 'Good afternoon',
        'evening': 'Good evening',
        'night': 'Good night'
    }

    # Get the current time of day
    timeOfDay = get_time_of_day()

    try:
        # Ensure timeOfDay is a valid key
        voicePackDataEntry = voicePackData[timeOfDay]
        print(f"Greeting for {timeOfDay}: {voicePackDataEntry}")
    except KeyError:
        print(f"Error: '{timeOfDay}' is not a valid key in voicePackData.")

    app.exec()
