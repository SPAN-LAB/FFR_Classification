import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QPlainTextEdit, QPushButton, QFileDialog, QLabel, QComboBox,
    QHBoxLayout, QScrollArea, QMessageBox, QTextEdit
)
from PyQt5.QtCore import Qt

PIPELINE_FILE = "pipeline.txt"

# Dummy function-to-options mapping (you can extend this)
FUNCTION_OPTIONS = {
    "clean_data": ["remove_nulls", "fill_mean", "drop_duplicates"],
    "normalize": ["min-max", "z-score", "none"],
    "extract_features": ["PCA", "LDA", "manual"],
    "train_model": ["SVM", "Random Forest", "Logistic Regression"],
}


class TextEditorTab(QWidget):
    def __init__(self, refresh_callback):
        super().__init__()
        self.refresh_callback = refresh_callback
        layout = QVBoxLayout()

        self.instructions = QLabel(
            "This text file is how you configure your preprocessing pipeline. To see detailed information about the builtin functions and what they expect as input and what they output, please see the Function Information tab.\n"
            "Builtin functions include: transpose_ffr, map_class_labels, trim_ffr, sub_average_ffr, test_split, k_folds_split"
            )        
        self.instructions.setWordWrap(True)
        layout.addWidget(self.instructions)

        self.editor = QPlainTextEdit()
        layout.addWidget(self.editor)

        btn_layout = QHBoxLayout()
        self.load_button = QPushButton("Load")
        self.save_button = QPushButton("Save")
        btn_layout.addWidget(self.load_button)
        btn_layout.addWidget(self.save_button)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        self.load_button.clicked.connect(self.load_file)
        self.save_button.clicked.connect(self.save_file)

        self.load_file()

    def load_file(self):
        if os.path.exists(PIPELINE_FILE):
            with open(PIPELINE_FILE, "r") as f:
                self.editor.setPlainText(f.read())
        else:
            self.editor.setPlainText("")

    def save_file(self):
        with open(PIPELINE_FILE, "w") as f:
            f.write(self.editor.toPlainText())
        QMessageBox.information(self, "Saved", "Pipeline file saved.")
        self.refresh_callback()


class FunctionBlock(QWidget):
    def __init__(self, name, index, move_up, move_down, delete):
        super().__init__()
        self.name = name
        self.index = index

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.label = QLabel(name)
        layout.addWidget(self.label)

        # Add dummy options (dropdown or text box)
        options = FUNCTION_OPTIONS.get(name.strip(), ["Option A", "Option B"])
        self.dropdown = QComboBox()
        self.dropdown.addItems(options)
        layout.addWidget(self.dropdown)

        # Move up/down/delete buttons
        self.up_button = QPushButton("↑")
        self.down_button = QPushButton("↓")
        self.delete_button = QPushButton("✕")

        layout.addWidget(self.up_button)
        layout.addWidget(self.down_button)
        layout.addWidget(self.delete_button)

        self.up_button.clicked.connect(lambda: move_up(self.index))
        self.down_button.clicked.connect(lambda: move_down(self.index))
        self.delete_button.clicked.connect(lambda: delete(self.index))


class ConfiguratorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.pipeline_steps = []
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)

        self.layout.addWidget(self.scroll_area)

        self.reload_button = QPushButton("Reload Pipeline")
        self.layout.addWidget(self.reload_button)
        self.reload_button.clicked.connect(self.load_pipeline)

        self.load_pipeline()

    def load_pipeline(self):
        # Clear previous
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        if not os.path.exists(PIPELINE_FILE):
            return

        with open(PIPELINE_FILE, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        self.pipeline_steps = lines

        for i, func_name in enumerate(self.pipeline_steps):
            block = FunctionBlock(
                func_name, i,
                move_up=self.move_up,
                move_down=self.move_down,
                delete=self.delete_step
            )
            self.scroll_layout.addWidget(block)

    def move_up(self, index):
        if index > 0:
            self.pipeline_steps[index], self.pipeline_steps[index - 1] = \
                self.pipeline_steps[index - 1], self.pipeline_steps[index]
            self.save_and_reload()

    def move_down(self, index):
        if index < len(self.pipeline_steps) - 1:
            self.pipeline_steps[index], self.pipeline_steps[index + 1] = \
                self.pipeline_steps[index + 1], self.pipeline_steps[index]
            self.save_and_reload()

    def delete_step(self, index):
        del self.pipeline_steps[index]
        self.save_and_reload()

    def save_and_reload(self):
        with open(PIPELINE_FILE, "w") as f:
            f.write("\n".join(self.pipeline_steps))
        self.load_pipeline()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Pipeline Editor")
        self.setGeometry(200, 200, 800, 500)

        self.tabs = QTabWidget()
        self.editor_tab = TextEditorTab(refresh_callback=self.refresh_config_tab)
        self.config_tab = ConfiguratorTab()

        self.tabs.addTab(self.editor_tab, "Pipeline Editor")
        self.tabs.addTab(self.config_tab, "Preprocessing Options")

        self.setCentralWidget(self.tabs)

    def refresh_config_tab(self):
        self.config_tab.load_pipeline()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
