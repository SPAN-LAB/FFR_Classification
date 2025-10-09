import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QComboBox,
    QHBoxLayout,
    QFrame, QListWidget, QListWidgetItem, QAbstractItemView, QLineEdit, QInputDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QFontDatabase

PIPELINE_FILE = "pipeline.txt"

# Global dark theme stylesheet
STYLE = """
QMainWindow { background-color: #121212; }
QLabel { color: #e6e6e6; }
QPushButton { background: #2b2b2b; color: #e6e6e6; border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px 10px; }
QPushButton:hover { background: #343434; }
QPushButton:disabled { background: #242424; color: #777777; border: 1px solid #2a2a2a; }
QComboBox { background: #1c1c1c; color: #e6e6e6; border: 1px solid #3a3a3a; border-radius: 6px; padding: 4px 8px; }
QFrame#FunctionCard { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px; }
"""

# Dummy function-to-options mapping (you can extend this)
FUNCTION_OPTIONS = {
    "clean_data": ["remove_nulls", "fill_mean", "drop_duplicates"],
    "normalize": ["min-max", "z-score", "none"],
    "extract_features": ["PCA", "LDA", "manual"],
    "train_model": ["SVM", "Random Forest", "Logistic Regression"],
}


class FunctionBlock(QFrame):
    def __init__(self, name, index, move_up, move_down, delete):
        super().__init__()
        self.name = name
        self.index = index

        self.setObjectName("FunctionCard")

        layout = QHBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        self.setLayout(layout)

        self.label = QLabel(name)
        layout.addWidget(self.label)

        # Add dummy options (dropdown or text box)
        options = FUNCTION_OPTIONS.get(name.strip(), ["Option A", "Option B"])
        self.dropdown = QComboBox()
        self.dropdown.addItems(options)
        self.dropdown.setMinimumWidth(160)
        self.dropdown.setToolTip(f"Options for {name}")
        layout.addWidget(self.dropdown)

        # Move up/down/delete buttons
        self.up_button = QPushButton("↑")
        self.down_button = QPushButton("↓")
        self.delete_button = QPushButton("✕")

        for b, tip in (
            (self.up_button, "Move up"),
            (self.down_button, "Move down"),
            (self.delete_button, "Remove step"),
        ):
            b.setFixedWidth(28)
            b.setToolTip(tip)

        layout.addWidget(self.up_button)
        layout.addWidget(self.down_button)
        layout.addWidget(self.delete_button)

        self.up_button.clicked.connect(lambda: move_up(self.index))
        self.down_button.clicked.connect(lambda: move_down(self.index))
        self.delete_button.clicked.connect(lambda: delete(self.index))
class DragHandleLabel(QLabel):
    def __init__(self, text: str = "≡", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.OpenHandCursor)

    def mousePressEvent(self, event):
        self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)
class FunctionItemWidget(QFrame):
    def __init__(self, name: str, arg: str = ""):
        super().__init__()
        self.setObjectName("FunctionCard")
        layout = QHBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        self.setLayout(layout)

        # Drag handle (three lines) and position number
        handle = DragHandleLabel("≡")
        handle.setFixedWidth(14)
        handle.setAlignment(Qt.AlignCenter)
        handle.setToolTip("Drag to reorder")
        layout.addWidget(handle)

        self.position_label = QLabel("1.")
        self.position_label.setFixedWidth(24)
        self.position_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.position_label)

        self.name = name
        self.label = QLabel(name)
        self.label.setMinimumWidth(120)
        layout.addWidget(self.label, 1)

        self.arg_edit = QLineEdit()
        self.arg_edit.setPlaceholderText("arguments (e.g. k=5, window=0:300)")
        self.arg_edit.setText(arg)
        self.arg_edit.setMinimumWidth(240)
        layout.addWidget(self.arg_edit, 2)

        # Delete button (red 'x') on the right
        self.delete_button = QPushButton("x")
        self.delete_button.setToolTip("Remove this step")
        self.delete_button.setFixedWidth(24)
        self.delete_button.setStyleSheet(
            "QPushButton { color: #ff5555; background: transparent; border: none; font-size: 14px; } "
            "QPushButton:hover { color: #ff7777; }"
        )
        self.delete_button.clicked.connect(self._request_delete)
        layout.addWidget(self.delete_button)

        # Emit change to enable autosave by parent
        self.arg_edit.textChanged.connect(self._notify_changed)

    def _notify_changed(self):
        # Parent container (ConfiguratorTab) will catch and save
        parent = self.parent()
        # QListWidget sets the widget inside a viewport, so we bubble to top-level
        while parent and not isinstance(parent, ConfiguratorTab):
            parent = parent.parent()
        if isinstance(parent, ConfiguratorTab):
            parent.notify_child_edit()

    def get_name(self) -> str:
        return self.name

    def get_arg(self) -> str:
        return self.arg_edit.text()

    def set_position(self, pos_index: int):
        # Display as 1-based index
        self.position_label.setText(f"{pos_index + 1}.")

    def _request_delete(self):
        parent = self.parent()
        while parent and not isinstance(parent, ConfiguratorTab):
            parent = parent.parent()
        if isinstance(parent, ConfiguratorTab):
            parent.remove_function_widget(self)



class ConfiguratorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.pipeline_steps = []
        self.pipeline_path = ""
        self.is_dirty = False
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(16, 12, 16, 12)
        self.layout.setSpacing(12)
        self.setLayout(self.layout)

        # Path row
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("pipeline path")
        self.open_create_button = QPushButton("Open")
        self.save_button_top = QPushButton("Save")
        self.save_button_top.hide()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.open_create_button)
        path_row.addWidget(self.save_button_top)
        self.layout.addLayout(path_row)

        self.open_create_button.clicked.connect(self.handle_open_or_create)
        self.save_button_top.clicked.connect(self.save_pipeline)
        self.path_edit.textChanged.connect(self._update_open_create_label)

        # Empty state label (shown when no file is open)
        self.empty_label = QLabel("No config file opened. Create one in the top bar.")
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.empty_label)

        # Draggable list of function blocks
        self.list_widget = QListWidget()
        self.list_widget.setObjectName("FunctionList")
        self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.setDefaultDropAction(Qt.MoveAction)
        self.list_widget.setSpacing(8)
        self.layout.addWidget(self.list_widget)

        # Initial visibility: no file open => show empty label, hide list
        self.empty_label.show()
        self.list_widget.hide()

        # List events
        self.list_widget.model().rowsMoved.connect(self._mark_dirty_and_autosave)
        self.list_widget.model().rowsMoved.connect(lambda *_: self._refresh_positions())

        # Initial state
        self._update_open_create_label()

        # Bottom bar: Add function
        bottom_row = QHBoxLayout()
        self.add_button = QPushButton("Add function")
        self.run_button = QPushButton("Run")
        bottom_row.addWidget(self.add_button)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self.run_button)
        self.layout.addLayout(bottom_row)
        self.add_button.clicked.connect(self.add_function)
        self.run_button.clicked.connect(self.run_pipeline)

        # Disable bottom buttons until a file is opened
        self._update_bottom_buttons_enabled()

    def load_pipeline(self):
        # Clear previous
        self.list_widget.clear()

        # No file selected/opened
        if not self.pipeline_path:
            self.path_edit.setText("")
            self.empty_label.show()
            self.list_widget.hide()
            self._update_bottom_buttons_enabled()
            return

        target_path = self.pipeline_path
        # Reflect path in the top text field
        self.path_edit.setText(target_path)

        if not os.path.exists(target_path):
            # No file exists at path yet
            self.empty_label.show()
            self.list_widget.hide()
            self._update_bottom_buttons_enabled()
            return

        with open(target_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        self.pipeline_steps = []

        for line in lines:
            parts = line.split()
            if len(parts) == 0:
                continue
            name = parts[0]
            arg = " ".join(parts[1:])
            self.pipeline_steps.append((name, arg))

            widget = FunctionItemWidget(name, arg)
            item = QListWidgetItem()
            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)

        # Update position labels
        self._refresh_positions()

        # Show list now that content is present
        self.empty_label.hide()
        self.list_widget.show()

        self.is_dirty = False
        self._update_save_visibility()
        self._update_bottom_buttons_enabled()
    def save_pipeline(self):
        lines = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            name = widget.get_name().strip()
            arg = widget.get_arg().strip()
            line = name if not arg else f"{name} {arg}"
            lines.append(line)
        target_path = self.pipeline_path or PIPELINE_FILE
        with open(target_path, "w") as f:
            f.write("\n".join(lines))
        self.is_dirty = False
        self._update_save_visibility()
        return True

    def handle_open_or_create(self):
        self.pipeline_path = self.path_edit.text().strip()
        if not self.pipeline_path:
            self.pipeline_path = PIPELINE_FILE
        if not os.path.exists(self.pipeline_path):
            # Create empty file for simplicity
            with open(self.pipeline_path, "w") as f:
                f.write("")
        self.load_pipeline()
        # Hide open/create until edits occur
        self.open_create_button.hide()
        self.save_button_top.hide()
        self._update_bottom_buttons_enabled()

    def _update_open_create_label(self):
        path = self.path_edit.text().strip()
        if path and not os.path.exists(path):
            self.open_create_button.setText("Create")
        else:
            self.open_create_button.setText("Open")
        # Enable buttons only when a non-empty path exists AND file exists or has been opened/created
        self._update_bottom_buttons_enabled()

    def _mark_dirty_and_autosave(self, *args, **kwargs):
        self.is_dirty = True
        self._update_save_visibility()
        self.save_pipeline()

    def _refresh_positions(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            if hasattr(widget, 'set_position'):
                widget.set_position(i)

    def run_pipeline(self):
        # Placeholder: integrate with execution layer later
        # For now, ensure positions are up-to-date and saved
        self._refresh_positions()
        self.save_pipeline()

    def notify_child_edit(self):
        self.is_dirty = True
        self._update_save_visibility()
        self.save_pipeline()

    def _update_save_visibility(self):
        # When editing starts, show Save button and hide open/create
        if self.is_dirty:
            self.open_create_button.hide()
            self.save_button_top.show()
        else:
            self.save_button_top.hide()
            self.open_create_button.show()

    def _update_bottom_buttons_enabled(self):
        if not hasattr(self, 'add_button') or not hasattr(self, 'run_button'):
            return
        has_path = bool(self.pipeline_path)
        enabled = has_path and os.path.exists(self.pipeline_path)
        self.add_button.setEnabled(enabled)
        self.run_button.setEnabled(enabled)

    def add_function(self):
        # Choose from FUNCTION_OPTIONS keys
        items = list(FUNCTION_OPTIONS.keys())
        if not items:
            return
        items.sort()
        selected, ok = QInputDialog.getItem(self, "Add function", "Select function:", items, 0, False)
        if not ok or not selected:
            return

        # Ensure a target path is defined
        if not self.pipeline_path:
            self.pipeline_path = self.path_edit.text().strip() or PIPELINE_FILE
            self.path_edit.setText(self.pipeline_path)

        # Ensure list is visible
        self.empty_label.hide()
        self.list_widget.show()

        # Append new item
        widget = FunctionItemWidget(selected, "")
        item = QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, widget)

        # Mark dirty and autosave
        self.is_dirty = True
        self._update_save_visibility()
        # Update numbering to reflect new item position
        self._refresh_positions()
        self.save_pipeline()

    def remove_function_widget(self, widget: QWidget):
        # Find the matching QListWidgetItem and remove it
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            w = self.list_widget.itemWidget(item)
            if w is widget:
                self.list_widget.takeItem(i)
                break
        # Refresh numbering and autosave
        self._refresh_positions()
        self.is_dirty = True
        self._update_save_visibility()
        self.save_pipeline()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FFR Pipeline Editor")
        self.setGeometry(200, 200, 800, 500)

        # Single, clean view: Configurator only
        self.config_tab = ConfiguratorTab()
        self.setCentralWidget(self.config_tab)

    def save_blocks_pipeline(self):
        ok = self.config_tab.save_pipeline()
        if ok:
            pass


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
