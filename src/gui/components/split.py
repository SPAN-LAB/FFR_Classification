from PyQt5.QtWidgets import QSplitter, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

from ..manager import Manager
from .stack import Stack


class Split(QSplitter):
    def __init__(self, parent, manager: Manager):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.manager = manager
        
        # Create left panel with Stack
        self.left_panel = QWidget()
        self.left_panel.setMinimumWidth(100)
        self.left_panel.setVisible(True)  # Explicitly show the left panel
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        
        self.stack = Stack(parent=self.left_panel, manager=self.manager)
        left_layout.addWidget(self.stack)
        
        # Create right panel (empty for now)
        self.right_panel = QWidget()
        self.right_panel.setStyleSheet("background-color: gray;")
        
        # Add panels to splitter
        self.addWidget(self.left_panel)
        self.addWidget(self.right_panel)
        
        # Set initial sizes (50/50 split)
        self.setSizes([400, 400])
        
        # Debug output after everything is set up
        print(f"Left panel visible: {self.left_panel.isVisible()}, geometry: {self.left_panel.geometry()}")
        print(f"Stack visible: {self.stack.isVisible()}, geometry: {self.stack.geometry()}")
        print(f"Splitter sizes: {self.sizes()}")
        print(f"Stack stylesheet: {self.stack.styleSheet()}")
        
        # Make the splitter handle visible and easy to drag
        self.setHandleWidth(4)
        
        # Make child widgets resizable
        self.setChildrenCollapsible(False)

