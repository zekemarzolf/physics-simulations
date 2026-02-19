import numpy as np
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QScrollArea, QWidget, QLabel, QPushButton, QFrame
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QPixmap
from PySide6 import QtWidgets
from StyleSheets import Styles

class CardWidget(QFrame):
    clicked = Signal()
    
    def __init__(self, width=300, height=400):
        super().__init__()
        
        self.setFixedSize(width, height)
        self.setCursor(Qt.PointingHandCursor)

        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(6)
        self.layout.setContentsMargins(8, 8, 8, 8)

        self.setObjectName("CardWidget")
        self.setAutoFillBackground(True)
        
    def mousePressEvent(self, event):
        self.clicked.emit()

class SimulationCard(CardWidget):
    def __init__(self, title, image_path):
        super().__init__()

        self.setObjectName("SimulationCard")
        
        
        self.image = QLabel()
        pixmap = QPixmap(image_path)
        self.image.setPixmap(
            pixmap.scaled(
                300, 300,
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
        self.resize(1280, 720)
        
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout()
        central.setLayout(layout)
        
        self.card1 = SimulationCard("2D N-Body Simulation", "null-image.jpg")
        self.card2 = SimulationCard("3D N-Body Simulation", "null-image.jpg")
        
        layout.addWidget(self.card1)
        layout.addWidget(self.card2)

app = QApplication(sys.argv)
app.setStyleSheet(Styles.MAIN_STYLESHEET)

window = MainWindow()
window.show()

app.exec()