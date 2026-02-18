import numpy as np
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QScrollArea, QWidget, QLabel, QPushButton
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QPixmap
from PySide6 import QtWidgets

class CardWidget(QWidget):
    clicked = Signal()
    
    def __init__(self, width=100, height=140):
        super().__init__()
        
        self.setFixedSize(width, height)
        self.setCursor(Qt.PointingHandCursor)

        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(6)
        self.layout.setContentsMargins(8, 8, 8, 8)

        self.setObjectName("CardWidget")

    def mousePressEvent(self, event):
        self.clicked.emit()

class ImageTitleCard(CardWidget):
    def __init__(self, title, image_path):
        super().__init__()

        self.image = QLabel()
        pixmap = QPixmap(image_path)
        self.image.setPixmap(
            pixmap.scaled(
                160, 90,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
        self.image.setAlignment(Qt.AlignCenter)

        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.image)
        self.layout.addWidget(self.label)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Physics Simulations")
        self.resize(1920, 1080)
        
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout()
        central.setLayout(layout)
        
        self.card1 = ImageTitleCard("hello world", "")
        self.card2 = ImageTitleCard("hi", "")
        
        layout.addWidget(self.card1)
        layout.addWidget(self.card2)

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()