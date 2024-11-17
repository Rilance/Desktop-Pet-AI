import os
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel
from PySide6.QtCore import Qt, QTimer, QPoint, QThread, Signal
from PySide6.QtGui import QFont, QMouseEvent, QScreen
from src.gui import main
import openai
import requests
from pydub import AudioSegment
from pydub.playback import play
import time
import threading
import datetime  # Newly added import statement

# Define API call function
class ApiCallThread(QThread):
    result = Signal(str)

    def __init__(self, input_text, parent=None):
        super(ApiCallThread, self).__init__(parent)
        self.input_text = input_text

    def run(self):
        result_text = call_api(self.input_text)
        self.result.emit(result_text)

# Define API call function
def call_api(input_text, retries=3, timeout=30):
    XAI_API_KEY = "xai-Zj8oBgRh6FMFI59dRTtwt0IlJftmzO7OEe7UidvKemVteLmvzegOCHI0BWRRbzNuBTS0m0pay7EuJeAf"  # Replace this with your API key
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
            # Print response content
            print(completion.choices[0].message)
            return completion.choices[0].message.content
        except openai.error.Timeout as e:
            print(f"Request timed out: {e}. Retrying...")
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
    return "API request timed out or an error occurred, please try again later."

# Define TTS call function
def call_tts_api(text, language):
    url = f"http://127.0.0.1:9880?text={text}&text_language={language}"
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if 'Content-Type' in response.headers and response.headers['Content-Type'] == 'audio/wav':
        # Save audio file
        with open("response.wav", "wb") as f:
            f.write(response.content)
        # Use pydub to play audio file
        audio = AudioSegment.from_file("response.wav", format="wav")
        play(audio)
    else:
        print("TTS API did not return a valid WAV file")

# Automatically detect language
def detect_language(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # Check if the character is a Chinese character
            return 'zh'
    return 'en'

# Create main window
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Send message to FireFly")
        self.setFixedSize(820, 300)  # Fixed window size
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 0, 10, 10)  # Adjust margins to move the input text box higher
        # Create input text box
        self.input_textbox = QTextEdit(self)
        self.input_textbox.setPlaceholderText("Enter text")
        layout.addWidget(self.input_textbox)
        # Create submit button
        self.submit_button = QPushButton("Submit", self)
        self.submit_button.clicked.connect(self.on_submit)
        layout.addWidget(self.submit_button)
        self.setLayout(layout)
    
        # Move the window to the center-bottom of the screen
        screen_geometry = QApplication.primaryScreen().availableGeometry()  # Get screen geometry
        screen_center = screen_geometry.center()  # Calculate screen center

        # Ensure the window size is already determined before moving
        self.resize(820, 300)  # Set window size, if needed
        self.move(
            screen_center.x() - self.frameGeometry().width() // 2,  # Use frameGeometry().width() to get accurate size
            screen_geometry.bottom() - self.height() - 30  # Place window closer to bottom
        )
        
        # Override mouse events to prevent moving
        def mousePressEvent(self, event):
            pass  # Do nothing, disables any mouse press handling for dragging

        def mouseMoveEvent(self, event):
            pass  # Do nothing, disables any mouse move handling for dragging
        
    def on_submit(self):
        input_text = self.input_textbox.toPlainText().strip()
        if input_text:
            self.api_thread = ApiCallThread(input_text)
            self.api_thread.result.connect(self.handle_response)
            self.api_thread.start()
            self.input_textbox.clear()  # Clear the input text box

    def handle_response(self, response_text):
        bubble_window.update_bubble(response_text)
        self.update_bubble_position()

    def update_bubble_position(self):
        # Iterate through all child widgets to find the QWidget object of the Firefly character image
        character_widget = None
        for widget in self.findChildren(QWidget):
            if isinstance(widget, QLabel):
                # Assume the Firefly character image is a QLabel
                character_widget = widget
                break
        if character_widget:
            character_pos = character_widget.pos()
            character_x = character_pos.x()
            character_y = character_pos.y()
            # Adjust the position of the dialog box to appear at the top left corner of Firefly
            bubble_x = character_x
            bubble_y = character_y
            bubble_window.move(bubble_x, bubble_y)

# Create answer bubble window class
class BubbleWindow(QWidget):
    def __init__(self, x, y):
        super().__init__()
        self.initUI(x, y)

    def initUI(self, x, y):
        self.setWindowTitle("Answer")  # Change the name of the output text box
        # Set the position and size of the bubble box
        self.setGeometry(x, y, 300, 100)
        # Create layout
        layout = QVBoxLayout()
        # Create answer label
        self.bubble_label = QLabel(self)
        self.bubble_label.setAlignment(Qt.AlignLeft)
        # Use default font, not PingFang font
        font = QFont()
        self.bubble_label.setFont(font)
        self.bubble_label.setWordWrap(True)  # Enable word wrap
        self.bubble_label.setStyleSheet("QLabel { background-color : white; color : black; border: 1px solid black; border-radius: 10px; padding: 10px; }")
        layout.addWidget(self.bubble_label)
        self.setLayout(layout)
        # Initially hide the answer bubble box
        self.hide()

    def update_bubble(self, response_text):
        if response_text.strip():
            self.bubble_label.setText("")  # 清空之前的文本
            self.show()

            # 确保 TTS 只调用一次
            def tts_thread(text, language):
                call_tts_api(text, language)
    
            # 语言检测
            language = detect_language(response_text)

            # 显示文本逐字加载效果
            for char in response_text:
                current_text = self.bubble_label.text()
                current_text += char
                self.bubble_label.setText(current_text)
                QApplication.processEvents()  # 刷新界面显示字符
                time.sleep(0.05)  # 控制显示速度

            self.bubble_label.adjustSize()

            # 仅在显示完文本后调用一次 TTS
            threading.Thread(target=tts_thread, args=(response_text, language)).start()
        else:
            self.hide()


    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.position().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.LeftButton:
            self.move(event.position().toPoint() - self.drag_position)
            event.accept()

# Determine the time of day based on the current time
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
