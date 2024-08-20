import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QFont, QMouseEvent
from src.gui import main
import requests
from pydub import AudioSegment
from pydub.playback import play

# 定义API调用函数
def call_api(input_text):
    url = "http://0.0.0.0:5000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-2-7b-chat.Q4_K_S.gguf",
        "messages": [{"role": "user", "content": input_text}],
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9
    }
    response = requests.post(url, headers=headers, json=data)
    
    # 打印响应内容
    print(response.text)
    
    # 尝试解析JSON
    try:
        return response.json()["choices"][0]["message"]["content"]
    except ValueError:
        return "API返回了无效的响应。"

# 定义TTS调用函数
def call_tts_api(text):
    url = "http://0.0.0.0:9880?text=" + text + "&text_language=en"
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    
    # 检查响应内容是否为有效的WAV文件
    if response.headers['Content-Type'] == 'audio/wav':
        # 保存音频文件
        with open("response.wav", "wb") as f:
            f.write(response.content)
        
        # 使用pydub播放音频文件
        audio = AudioSegment.from_file("response.wav", format="wav")
        play(audio)
    else:
        print("TTS API 返回的不是有效的 WAV 文件")

# 创建主窗口
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("MyFlowingFireflyWife with Text Generation")

        # 创建布局
        layout = QVBoxLayout()

        # 创建输入文本框
        self.input_textbox = QTextEdit(self)
        self.input_textbox.setPlaceholderText("输入文本")
        layout.addWidget(self.input_textbox)

        # 创建提交按钮
        self.submit_button = QPushButton("提交", self)
        self.submit_button.clicked.connect(self.on_submit)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)

    def on_submit(self):
        input_text = self.input_textbox.toPlainText().strip()
        if input_text:
            response_text = call_api(input_text)
            bubble_window.update_bubble(response_text)
            self.update_bubble_position()
            call_tts_api(response_text)  # 调用TTS API并播放语音

    def update_bubble_position(self):
        # 获取流萤人物形象的位置
        character_widget = self.findChild(QWidget, "character_widget")  # 假设流萤人物形象的QWidget对象名为"character_widget"
        if character_widget:
            character_x = character_widget.x()
            character_y = character_widget.y()
            character_height = character_widget.height()
            
            # 调整对话框的位置，使其出现在流萤的下方
            bubble_x = character_x
            bubble_y = character_y + character_height
            
            bubble_window.move(bubble_x, bubble_y)

# 创建回答气泡框窗口类
class BubbleWindow(QWidget):
    def __init__(self, x, y):
        super().__init__()
        self.initUI(x, y)

    def initUI(self, x, y):
        self.setWindowTitle("Response Bubble")
        
        # 设置气泡框的位置和大小
        self.setGeometry(x, y, 300, 100)
        
        # 创建布局
        layout = QVBoxLayout()
        
        # 创建回答标签
        self.bubble_label = QLabel(self)
        self.bubble_label.setAlignment(Qt.AlignLeft)
        self.bubble_label.setStyleSheet("QLabel { background-color : white; color : black; border: 1px solid black; border-radius: 10px; padding: 10px; font-family: 'Microsoft YaHei'; }")
        
        layout.addWidget(self.bubble_label)
        
        self.setLayout(layout)
        
        # 初始隐藏回答气泡框
        self.hide()

    def update_bubble(self, response_text):
        if response_text.strip():
            self.bubble_label.setText(response_text)
            self.bubble_label.adjustSize()
            self.show()
        else:
            self.hide()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

if __name__ == '__main__':
    # gui init
    main.logger.add("log\\latest.log", rotation="500 MB")
    app = QApplication(sys.argv)
    window = main.MainWindow(app)
    window.show()

    # 创建并显示主窗口
    text_gen_window = MainWindow()
    text_gen_window.show()

    # 创建并显示回答气泡框窗口，初始位置可以自定义（例如：x=300，y=200）
    bubble_window = BubbleWindow(300, 200)
    bubble_window.show()

    app.exec()
