from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QSplitter, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy
import sys

from .components.toolbar import Toolbar
from .components.top_bar_info import TopBarInfo
from .components.function_builder import FunctionBuilder
from .components.bottom_bar import BottomBar
from .components.function_block import FunctionBlock

from .manager import Manager

class SplitterExample(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QSplitter Example")
        self.resize(400, 200)

        # Create a horizontal splitter
        splitter = QSplitter(Qt.Horizontal)

        # Add left and right widgets
        left_label = QLabel("Hello")
        left_label.setAlignment(Qt.AlignCenter)
        right_label = QLabel("World")
        right_label.setAlignment(Qt.AlignCenter)

        splitter.addWidget(left_label)
        splitter.addWidget(right_label)

        # Put the splitter in a layout
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.manager = Manager()

        # Setup window
        self.setWindowTitle("FFR Pipeline")
        self.setGeometry(100, 100, 800, 600)

        # Set the layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        self.toolbar = Toolbar("Toolbar", parent=self, manager=self.manager)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        # TopBarInfo
        self.top_bar_info = TopBarInfo(parent=self, manager=self.manager)
        layout.addWidget(self.top_bar_info, alignment=Qt.AlignTop)


        # Main content
        self.main_content = QSplitter(parent=self)
        self.main_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_content.setContentsMargins(0, 0, 0, 0)  # No margins
        self.builder = FunctionBuilder(parent=self, manager=self.manager)
        self.builder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        right = QLabel("Right", parent=self)
        right.setAlignment(Qt.AlignCenter)
        self.main_content.addWidget(self.builder)
        self.main_content.addWidget(right)
        layout.addWidget(self.main_content)

        # Bottom bar 
        self.bottom_bar = BottomBar(parent=self, manager=self.manager)
        layout.addWidget(self.bottom_bar, alignment=Qt.AlignBottom)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
