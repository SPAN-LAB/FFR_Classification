from PyQt5.QtWidgets import QWidget, QSizePolicy

from ..manager import Manager


class Stack(QWidget):
    def __init__(self, parent, manager: Manager):
        super().__init__(parent)
        self.manager = manager
        
        # Set black background
        self.setAutoFillBackground(True)
        self.setStyleSheet("background-color: black; border: 3px solid red;")
        
        # Set minimum size so it's visible
        self.setMinimumSize(100, 100)
        
        # Make sure it expands to fill available space
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Make sure it's visible
        self.setVisible(True)

