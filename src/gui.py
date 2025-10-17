import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QComboBox,
    QHBoxLayout,
    QFrame, QListWidget, QListWidgetItem, QAbstractItemView, QLineEdit, QInputDialog, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QFontDatabase
from FriendlyFunction import FriendlyFunction
from FriendlyFunctionManager import FriendlyFunctionManager

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


class FunctionBlock(QFrame):
    """
    Legacy placeholder retained for compatibility with older code; not used in current UI.
    """

    def __init__(self, name, index, move_up, move_down, delete):
        super().__init__()
        # Legacy placeholder retained for compatibility; no longer used.
        self.setObjectName("FunctionCard")


class DragHandleView(QLabel):
    """
    Small label that acts as a drag handle; shows grab/closed-hand cursors on hover/press.
    """

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


class FunctionBlockView(QFrame):
    """
    Visual representation of a pipeline step: handle, position number, name, args, and delete action.
    """

    def __init__(self, name: str, arg: str = "", guidance: dict | None = None):
        super().__init__()
        self.setObjectName("FunctionCard")
        layout = QHBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        self.setLayout(layout)

        # Drag handle (three lines) and position number
        handle = DragHandleView("≡")
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

        # Parameter views row: generated from guidance (param -> type or schema dict)
        self.param_views = {}
        guidance = guidance or {}
        params_row = QHBoxLayout()
        params_row.setSpacing(6)
        for param_name, spec in guidance.items():
            # spec can be a type or a dict with keys: 'type', 'default'
            if isinstance(spec, dict):
                param_type = spec.get('type', str)
                default_value = spec.get('default', None)
            else:
                param_type = spec
                default_value = None

            param_label = QLabel(f"{param_name}:")
            params_row.addWidget(param_label)
            if param_type is int:
                editor = QSpinBox()
                editor.setMinimum(-10_000_000)
                editor.setMaximum(10_000_000)
                editor.setFixedWidth(72)
                if default_value is not None:
                    try:
                        editor.setValue(int(default_value))
                    except Exception:
                        pass
                editor.valueChanged.connect(self._notify_changed)
            elif param_type is float:
                editor = QDoubleSpinBox()
                editor.setDecimals(6)
                editor.setMinimum(-1e12)
                editor.setMaximum(1e12)
                editor.setFixedWidth(96)
                if default_value is not None:
                    try:
                        editor.setValue(float(default_value))
                    except Exception:
                        pass
                editor.valueChanged.connect(self._notify_changed)
            else:
                editor = QLineEdit()
                editor.setFixedWidth(140)
                if default_value is not None:
                    editor.setText(str(default_value))
                editor.textChanged.connect(self._notify_changed)
            self.param_views[param_name] = (editor, param_type)
            params_row.addWidget(editor)
        layout.addLayout(params_row, 2)

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

        # No single arg edit; param editors above signal changes

    def _notify_changed(self):
        # Parent container (BuildFunctionView) will catch and save
        parent = self.parent()
        # QListWidget sets the widget inside a viewport, so we bubble to top-level
        while parent and not isinstance(parent, BuildFunctionView):
            parent = parent.parent()
        if isinstance(parent, BuildFunctionView):
            parent.notify_child_edit()

    def get_name(self) -> str:
        return self.name

    def get_arg(self) -> str:
        # Reconstruct key=value string for backward-compat save format
        parts = []
        for key, (editor, typ) in self.param_views.items():
            if typ is int:
                parts.append(f"{key}={int(editor.value())}")
            elif typ is float:
                parts.append(f"{key}={float(editor.value())}")
            else:
                parts.append(f"{key}={editor.text()}")
        return " ".join(parts)

    def get_args_dict(self) -> dict:
        args = {}
        for key, (editor, typ) in self.param_views.items():
            if typ is int:
                args[key] = int(editor.value())
            elif typ is float:
                args[key] = float(editor.value())
            else:
                args[key] = editor.text()
        return args

    def apply_args_dict(self, args: dict) -> None:
        for key, (editor, typ) in self.param_views.items():
            if key not in args:
                continue
            value = args[key]
            try:
                if typ is int:
                    editor.setValue(int(value))
                elif typ is float:
                    editor.setValue(float(value))
                else:
                    editor.setText(str(value))
            except Exception:
                # Ignore bad values; keep current
                pass

    def set_supported(self, is_supported: bool) -> None:
        # Red if unsupported, normal otherwise
        if is_supported:
            self.label.setStyleSheet("")
        else:
            self.label.setStyleSheet("QLabel { color: #ff5555; }")

    def set_position(self, pos_index: int):
        # Display as 1-based index
        self.position_label.setText(f"{pos_index + 1}.")

    def _request_delete(self):
        parent = self.parent()
        while parent and not isinstance(parent, BuildFunctionView):
            parent = parent.parent()
        if isinstance(parent, BuildFunctionView):
            parent.remove_function_widget(self)


class BuildFunctionView(QWidget):
    """
    Main builder view for configuring the pipeline: path input, DnD function list,
    add/run controls, empty state messaging, and autosave behavior.
    """

    def __init__(self):
        super().__init__()
        self.pipeline_steps = []
        self.pipeline_path = ""
        self.is_dirty = False
        # Initialize manager (subject is loaded via 'load_data')
        self.function_manager = FriendlyFunctionManager()
        
        self.init_ui()

    def init_ui(self):
        self.layout_view = QVBoxLayout()
        self.layout_view.setContentsMargins(16, 12, 16, 12)
        self.layout_view.setSpacing(12)
        self.setLayout(self.layout_view)

        # Top bar: filename (left), Close and Save (next), Open (far right when no file open)
        top_row = QHBoxLayout()
        self.filename_label_view = QLabel("")
        self.filename_label_view.setMinimumWidth(0)
        self.close_and_save_button_view = QPushButton("Close and Save")
        self.new_button_view = QPushButton("New Pipeline")
        self.open_button_view = QPushButton("Open")

        top_row.addWidget(self.filename_label_view)
        top_row.addWidget(self.close_and_save_button_view)
        top_row.addStretch(1)
        top_row.addWidget(self.new_button_view)
        top_row.addWidget(self.open_button_view)
        self.layout_view.addLayout(top_row)

        self.close_and_save_button_view.clicked.connect(self.close_and_save)
        self.new_button_view.clicked.connect(self.new_pipeline)
        self.open_button_view.clicked.connect(self.open_file_dialog)

        # Empty state label (shown when no file is open)
        self.empty_label_view = QLabel("No config file opened. Create one in the top bar.")
        self.empty_label_view.setWordWrap(True)
        self.empty_label_view.setAlignment(Qt.AlignCenter)
        self.layout_view.addWidget(self.empty_label_view)

        # Draggable list of function blocks
        self.list_view = QListWidget()
        self.list_view.setObjectName("FunctionList")
        self.list_view.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_view.setDefaultDropAction(Qt.MoveAction)
        self.list_view.setSpacing(8)
        self.layout_view.addWidget(self.list_view)

        # Initial visibility: no file open => show empty label, hide list, show Open
        self.empty_label_view.show()
        self.list_view.hide()
        self._update_top_controls()

        # List events
        self.list_view.model().rowsMoved.connect(self._mark_dirty_and_autosave)
        self.list_view.model().rowsMoved.connect(lambda *_: self._refresh_positions())

        # Initial state
        self._update_bottom_buttons_enabled()

        # Bottom bar: Add function
        bottom_row = QHBoxLayout()
        self.add_button_view = QPushButton("Add function")
        self.run_button_view = QPushButton("Run")
        bottom_row.addWidget(self.add_button_view)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self.run_button_view)
        self.layout_view.addLayout(bottom_row)
        self.add_button_view.clicked.connect(self.add_function)
        self.run_button_view.clicked.connect(self.run_pipeline)

        # Disable bottom buttons until a file is opened
        self._update_bottom_buttons_enabled()

    def load_pipeline(self):
        # Clear previous
        self.list_view.clear()

        # No file selected/opened
        if not self.pipeline_path:
            self.empty_label_view.show()
            self.list_view.hide()
            self._update_top_controls()
            self._update_bottom_buttons_enabled()
            return

        target_path = self.pipeline_path
        # Reflect filename in the top bar
        self.filename_label_view.setText(os.path.basename(target_path))

        if not os.path.exists(target_path):
            # No file exists at path yet
            self.empty_label_view.show()
            self.list_view.hide()
            self._update_top_controls()
            self._update_bottom_buttons_enabled()
            return

        with open(target_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        self.pipeline_steps = []

        for line in lines:
            parts = line.split()
            if len(parts) == 0:
                continue
            # Reconstruct full name (can contain spaces) by splitting before first key=value token
            first_kv_index = None
            for idx, token in enumerate(parts):
                if '=' in token:
                    first_kv_index = idx
                    break
            if first_kv_index is None:
                # Entire line is the function name; no args
                name = line
                arg = ""
            else:
                name = " ".join(parts[:first_kv_index])
                arg = " ".join(parts[first_kv_index:])
            self.pipeline_steps.append((name, arg))

            usage_schema = self._get_usage_schema(name) or {}
            # Hide args if the function consumes EEGData only
            if self._get_source_type(name) == FriendlyFunction.SourceType.EEG_DATA:
                usage_schema = {}
            widget = FunctionBlockView(name, arg, usage_schema)
            item = QListWidgetItem()
            item.setSizeHint(widget.sizeHint())
            self.list_view.addItem(item)
            self.list_view.setItemWidget(item, widget)
            # Apply saved args to override defaults
            if arg:
                widget.apply_args_dict(self._parse_named_args(arg))

        # Update position labels
        self._refresh_positions()
        self._refresh_support_highlighting()

        # Show list now that content is present
        self.empty_label_view.hide()
        self.list_view.show()

        self.is_dirty = False
        self._update_top_controls()
        self._update_bottom_buttons_enabled()
        self._refresh_support_highlighting()
    
    def save_pipeline(self):
        lines = []
        for i in range(self.list_view.count()):
            item = self.list_view.item(i)
            widget = self.list_view.itemWidget(item)
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

    def open_file_dialog(self):
        # Use native dialog to select a file to open
        file_path, _ = QFileDialog.getOpenFileName(self, "Open pipeline", os.getcwd(), "Text Files (*.txt);;All Files (*)")
        if not file_path:
            return
        self.pipeline_path = file_path
        if not os.path.exists(self.pipeline_path):
            with open(self.pipeline_path, "w") as f:
                f.write("")
        self.load_pipeline()
        self._update_top_controls()
        self._update_bottom_buttons_enabled()

    def new_pipeline(self):
        # Use native dialog to choose where to create a new pipeline file
        file_path, _ = QFileDialog.getSaveFileName(self, "New pipeline", os.path.join(os.getcwd(), "pipeline.txt"), "Text Files (*.txt);;All Files (*)")
        if not file_path:
            return
        # Ensure .txt extension if none provided
        if not os.path.splitext(file_path)[1]:
            file_path = file_path + ".txt"
        # Create/overwrite empty file
        with open(file_path, "w") as f:
            f.write("")
        self.pipeline_path = file_path
        self.load_pipeline()
        self._update_top_controls()
        self._update_bottom_buttons_enabled()

    def close_and_save(self):
        # Save current pipeline, then close and reset UI to initial state
        if self.pipeline_path:
            self.save_pipeline()
        self.pipeline_path = ""
        self.list_view.clear()
        self.empty_label_view.show()
        self.list_view.hide()
        self.is_dirty = False
        self._update_top_controls()
        self._update_bottom_buttons_enabled()

    def _mark_dirty_and_autosave(self, *args, **kwargs):
        self.is_dirty = True
        self._update_save_visibility()
        self.save_pipeline()

    def _refresh_positions(self):
        for i in range(self.list_view.count()):
            item = self.list_view.item(i)
            widget = self.list_view.itemWidget(item)
            if hasattr(widget, 'set_position'):
                widget.set_position(i)

    def run_pipeline(self):
        # Execute functions sequentially via manager; dynamically resolves functions as they become available
        for i in range(self.list_view.count()):
            item = self.list_view.item(i)
            widget = self.list_view.itemWidget(item)
            name = widget.get_name().strip()
            # Check function availability at this moment
            if name not in set(self._get_all_function_names()):
                QMessageBox.warning(self, "Unsupported functions", f"Cannot run. Unsupported: {name}")
                return
            # Prefer structured args when available
            if hasattr(widget, 'get_args_dict'):
                args_dict = widget.get_args_dict()
            else:
                args_text = widget.get_arg().strip()
                args_dict = self._parse_named_args(args_text)
            try:
                self.function_manager.run_function(name, **args_dict)
            except Exception as e:
                QMessageBox.warning(self, "Execution error", f"Failed to run {name}: {e}")
                return

        # Ensure positions are up-to-date and saved
        self._refresh_positions()
        self.save_pipeline()

    def notify_child_edit(self):
        self.is_dirty = True
        self._update_top_controls()
        self.save_pipeline()

    def _update_save_visibility(self):
        # Deprecated in top-bar refactor; retained for compatibility if called
        self._update_top_controls()

    def _update_bottom_buttons_enabled(self):
        if not hasattr(self, 'add_button_view') or not hasattr(self, 'run_button_view'):
            return
        has_path = bool(self.pipeline_path)
        enabled = has_path and os.path.exists(self.pipeline_path)
        self.add_button_view.setEnabled(enabled)
        self.run_button_view.setEnabled(enabled)

    def _update_top_controls(self):
        # Show filename and Close/Save when a file is open; otherwise show Open button only
        file_open = bool(self.pipeline_path)
        if hasattr(self, 'filename_label_view'):
            self.filename_label_view.setVisible(file_open)
            if file_open and os.path.exists(self.pipeline_path):
                self.filename_label_view.setText(os.path.basename(self.pipeline_path))
            elif file_open:
                self.filename_label_view.setText(os.path.basename(self.pipeline_path))
            else:
                self.filename_label_view.setText("")
        if hasattr(self, 'close_and_save_button_view'):
            self.close_and_save_button_view.setVisible(file_open)
        if hasattr(self, 'open_button_view'):
            self.open_button_view.setVisible(not file_open)
        if hasattr(self, 'new_button_view'):
            self.new_button_view.setVisible(not file_open)

    def _refresh_support_highlighting(self):
        available_names = set(self._get_all_function_names())
        for i in range(self.list_view.count()):
            item = self.list_view.item(i)
            widget = self.list_view.itemWidget(item)
            name = widget.get_name().strip()
            is_supported = name in available_names
            if hasattr(widget, 'set_supported'):
                widget.set_supported(is_supported)

    def _parse_named_args(self, args_text: str) -> dict:
        result = {}
        if not args_text:
            return result
        parts = args_text.split()
        for part in parts:
            if '=' not in part:
                continue
            key, value = part.split('=', 1)
            value_cast = self._auto_cast(value)
            result[key] = value_cast
        return result

    def _auto_cast(self, value: str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def add_function(self):
        # Choose from manager-registered function names
        items = self._get_all_function_names()
        items.sort()
        selected, ok = QInputDialog.getItem(self, "Add function", "Select function:", items, 0, False)
        if not ok or not selected:
            return

        # Ensure list is visible
        self.empty_label_view.hide()
        self.list_view.show()

        # Append new item using usage as guidance
        usage_schema = self._get_usage_schema(selected) or {}
        widget = FunctionBlockView(selected, "", usage_schema)
        item = QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        self.list_view.addItem(item)
        self.list_view.setItemWidget(item, widget)

        # Mark dirty and autosave
        self.is_dirty = True
        self._update_save_visibility()
        # Update numbering and support highlighting
        self._refresh_positions()
        self._refresh_support_highlighting()
        self.save_pipeline()

    def _get_all_function_names(self):
        return [f.name for f in getattr(self.function_manager, 'possible_functions', [])]

    def _get_usage_schema(self, function_name: str):
        # Build a simple schema mapping parameter_name -> { 'type': type, 'default': value }
        for f in getattr(self.function_manager, 'possible_functions', []):
            if f.name == function_name:
                schema = {}
                if hasattr(f, 'user_facing_arguments'):
                    for arg in f.user_facing_arguments():
                        schema[arg.parameter_name] = {
                            'type': arg.data_type,
                            'default': arg.default_value,
                        }
                return schema
        return {}

    def remove_function_widget(self, widget: QWidget):
        # Find the matching QListWidgetItem and remove it
        for i in range(self.list_view.count()):
            item = self.list_view.item(i)
            w = self.list_view.itemWidget(item)
            if w is widget:
                self.list_view.takeItem(i)
                break
        # Refresh numbering and autosave
        self._refresh_positions()
        self._refresh_support_highlighting()
        self.is_dirty = True
        self._update_top_controls()
        self.save_pipeline()


class MainWindowView(QMainWindow):
    """
    Application main window hosting the BuildFunctionView as the central widget.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Frequency-Following Response Pipeline Editor")
        self.setGeometry(200, 200, 800, 500)

        # Single, clean view: Configurator only
        self.build_function_view = BuildFunctionView()
        self.setCentralWidget(self.build_function_view)

    def save_blocks_pipeline(self):
        ok = self.build_function_view.save_pipeline()
        if ok:
            pass


def main():
    app = QApplication(sys.argv)
    window = MainWindowView()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
