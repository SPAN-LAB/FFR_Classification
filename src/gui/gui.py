from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QLineEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .manager import Manager
from ..core import PipelineState
from ..core.utils.function_detail import ArgumentDetail, FunctionDetail, Selection


def _current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _function_detail(bound_function: Callable) -> Optional[FunctionDetail]:
    func = getattr(bound_function, "__func__", bound_function)
    return getattr(func, "detail", None)


class ArgumentEditor(QWidget):
    """Widget wrapper that knows how to extract a typed value."""

    def __init__(self, detail: ArgumentDetail, manager: Manager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._detail = detail
        self._manager = manager
        self._getter: Callable[[], Any]
        self._widget: QWidget
        self._build()

    @property
    def label(self) -> str:
        return self._detail.label

    @property
    def widget(self) -> QWidget:
        return self._widget

    def value(self) -> Any:
        return self._getter()

    # Builders -----------------------------------------------------------------

    def _build(self) -> None:
        typ = self._detail.type
        if typ is int:
            editor = QSpinBox()
            editor.setMinimum(-10_000_000)
            editor.setMaximum(10_000_000)
            if self._detail.default_value is not None:
                editor.setValue(int(self._detail.default_value))
            self._widget = editor
            self._getter = lambda: int(editor.value())
        elif typ is float:
            editor = QDoubleSpinBox()
            editor.setDecimals(6)
            editor.setMinimum(-1e9)
            editor.setMaximum(1e9)
            if self._detail.default_value is not None:
                editor.setValue(float(self._detail.default_value))
            self._widget = editor
            self._getter = lambda: float(editor.value())
        elif typ is str:
            editor = QLineEdit()
            if self._detail.default_value is not None:
                editor.setText(str(self._detail.default_value))
            self._widget = editor
            self._getter = lambda: editor.text()
        elif typ is dict:
            editor = QPlainTextEdit()
            editor.setPlaceholderText("Enter JSON object")
            if isinstance(self._detail.default_value, dict):
                editor.setPlainText(json.dumps(self._detail.default_value, indent=2))
            self._widget = editor

            def getter() -> dict[str, Any]:
                text = editor.toPlainText().strip()
                if not text:
                    return {}
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self._detail.label}: invalid JSON ({exc})") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{self._detail.label}: expected JSON object")
                return value

            self._getter = getter
        elif typ is Selection:
            selection = self._resolve_selection()
            combo = QComboBox()
            combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            mapping = self._selection_mapping(selection)
            for option in mapping.keys():
                combo.addItem(option)
            if combo.count():
                combo.setCurrentIndex(0)
            self._widget = combo

            def getter() -> Any:
                current = combo.currentText()
                mapping = self._selection_mapping(selection)
                if current not in mapping:
                    raise ValueError(f"{self._detail.label}: '{current}' is not available")
                return mapping[current]

            self._getter = getter
        else:
            editor = QLineEdit()
            if self._detail.default_value is not None:
                editor.setText(str(self._detail.default_value))
            self._widget = editor
            self._getter = lambda: editor.text()

    def _resolve_selection(self) -> Selection:
        default = self._detail.default_value
        if isinstance(default, Selection):
            return default
        raise ValueError(
            f"{self._detail.label}: Selection arguments require a Selection default value."
        )

    def _selection_mapping(self, selection: Selection) -> dict[str, Any]:
        mapping_fn = selection.option_value_map
        try:
            return mapping_fn(self._manager.state)
        except TypeError:
            return mapping_fn()


class FunctionDetailPanel(QWidget):
    run_requested = pyqtSignal(str, dict)
    queue_requested = pyqtSignal(str, dict)

    def __init__(self, manager: Manager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._current_name: Optional[str] = None
        self._current_detail: Optional[FunctionDetail] = None
        self._editors: list[ArgumentEditor] = []

        self._title = QLabel("Select a function to view details.")
        self._title.setWordWrap(True)
        self._description = QLabel("")
        self._description.setWordWrap(True)

        self._form_container = QWidget()
        self._form_layout = QFormLayout()
        self._form_layout.setLabelAlignment(Qt.AlignRight)
        self._form_container.setLayout(self._form_layout)

        self._run_button = QPushButton("Run")
        self._queue_button = QPushButton("Add to Queue")
        self._run_button.clicked.connect(self._emit_run)
        self._queue_button.clicked.connect(self._emit_queue)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self._run_button)
        button_row.addWidget(self._queue_button)

        layout = QVBoxLayout()
        layout.addWidget(self._title)
        layout.addWidget(self._description)
        layout.addWidget(self._form_container)
        layout.addLayout(button_row)
        layout.addStretch(1)
        self.setLayout(layout)
        self._set_enabled(False)

    # Public API ---------------------------------------------------------------

    def display_function(self, name: str, function: Callable) -> None:
        self._clear_form()
        detail = _function_detail(function)
        if detail is None:
            self._title.setText(name)
            self._description.setText("This function is not available in the GUI.")
            self._set_enabled(False)
            return

        self._current_name = name
        self._current_detail = detail
        self._title.setText(detail.label or name)
        self._description.setText(detail.description or "")

        for argument in detail.argument_details:
            editor = ArgumentEditor(argument, self._manager, self)
            self._editors.append(editor)
            self._form_layout.addRow(f"{argument.label}:", editor.widget)

        self._set_enabled(True)

    def clear(self) -> None:
        self._clear_form()
        self._set_enabled(False)
        self._title.setText("Select a function to view details.")
        self._description.setText("")

    # Helpers -----------------------------------------------------------------

    def _emit_run(self) -> None:
        if not self._current_name:
            return
        try:
            params = self._collect_parameters()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self.run_requested.emit(self._current_name, params)

    def _emit_queue(self) -> None:
        if not self._current_name:
            return
        try:
            params = self._collect_parameters()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self.queue_requested.emit(self._current_name, params)

    def _collect_parameters(self) -> dict:
        params: Dict[str, Any] = {}
        for editor in self._editors:
            params[editor.label] = editor.value()
        return params

    def _clear_form(self) -> None:
        self._current_name = None
        self._current_detail = None
        for i in reversed(range(self._form_layout.count())):
            item = self._form_layout.takeAt(i)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._editors.clear()

    def _set_enabled(self, enabled: bool) -> None:
        self._run_button.setEnabled(enabled)
        self._queue_button.setEnabled(enabled)
        self._form_container.setEnabled(enabled)

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Invalid parameters", message)


class QueuePanel(QWidget):
    remove_requested = pyqtSignal(int)
    clear_requested = pyqtSignal()
    run_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)

        self.run_button = QPushButton("Run Queue")
        self.remove_button = QPushButton("Remove Selected")
        self.clear_button = QPushButton("Clear Queue")

        self.run_button.clicked.connect(self.run_requested.emit)
        self.remove_button.clicked.connect(self._emit_remove)
        self.clear_button.clicked.connect(self.clear_requested.emit)

        button_row = QHBoxLayout()
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.remove_button)
        button_row.addWidget(self.clear_button)
        button_row.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Queued Functions"))
        layout.addWidget(self.list_widget)
        layout.addLayout(button_row)
        self.setLayout(layout)

    def add_item(self, display_text: str) -> None:
        self.list_widget.addItem(display_text)

    def remove_selected(self) -> int:
        row = self.list_widget.currentRow()
        if row >= 0:
            item = self.list_widget.takeItem(row)
            del item
        return row

    def clear(self) -> None:
        self.list_widget.clear()

    def _emit_remove(self) -> None:
        row = self.list_widget.currentRow()
        if row >= 0:
            self.remove_requested.emit(row)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FFR Pipeline Manager")
        self.resize(1100, 700)

        self.manager = Manager()
        self.function_map: dict[str, Callable] = {}

        self._create_widgets()
        self._wire_signals()
        self._refresh_functions()
        self._update_summary()

    # UI creation --------------------------------------------------------------

    def _create_widgets(self) -> None:
        load_button = QPushButton("Load Subjects…")
        load_button.clicked.connect(self._choose_subjects)

        restore_button = QPushButton("Restore Subjects")
        restore_button.clicked.connect(self._restore_subjects)

        toolbar = QHBoxLayout()
        toolbar.addWidget(load_button)
        toolbar.addWidget(restore_button)
        toolbar.addStretch(1)

        self.function_list = QListWidget()
        self.function_list.setMinimumWidth(250)

        self.detail_panel = FunctionDetailPanel(self.manager)
        self.queue_panel = QueuePanel()

        left = QVBoxLayout()
        left.addWidget(QLabel("Functions"))
        left.addWidget(self.function_list)

        left_widget = QWidget()
        left_widget.setLayout(left)

        self.summary_label = QLabel("No subjects loaded.")
        self.summary_label.setWordWrap(True)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Logs appear here.")

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.detail_panel)
        right_layout.addWidget(self.queue_panel)
        right_layout.addWidget(QLabel("Subject Summary"))
        right_layout.addWidget(self.summary_label)
        right_layout.addWidget(QLabel("Log"))
        right_layout.addWidget(self.log_view)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter = QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        central_layout = QVBoxLayout()
        central_layout.addLayout(toolbar)
        central_layout.addWidget(splitter)

        container = QWidget()
        container.setLayout(central_layout)
        self.setCentralWidget(container)

    def _wire_signals(self) -> None:
        self.function_list.currentItemChanged.connect(self._on_function_selected)
        self.detail_panel.run_requested.connect(self._handle_run_request)
        self.detail_panel.queue_requested.connect(self._handle_queue_request)
        self.queue_panel.remove_requested.connect(self._remove_queue_item)
        self.queue_panel.clear_requested.connect(self._clear_queue)
        self.queue_panel.run_requested.connect(self._run_queue)

    # Actions ------------------------------------------------------------------

    def _choose_subjects(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select subject folder", str(Path.cwd()))
        if not folder:
            return
        try:
            self.manager.load_subjects(folder)
        except Exception as exc:
            self._log_exception("Failed to load subjects", exc)
            QMessageBox.critical(self, "Load error", str(exc))
            return
        self.append_log(f"Loaded subjects from {folder}")
        self.queue_panel.clear()
        self.manager.functions.clear()
        self._refresh_functions()
        self._update_summary()

    def _restore_subjects(self) -> None:
        try:
            self.manager.state = PipelineState()
            self.manager.initial_subjects_state.save(to=self.manager.state)
        except Exception as exc:
            self._log_exception("Failed to restore subjects", exc)
            QMessageBox.critical(self, "Restore error", str(exc))
            return
        self.append_log("Restored subjects to the initial snapshot.")
        self._refresh_functions()
        self._update_summary()

    def _on_function_selected(self, current: QListWidgetItem) -> None:
        if current is None:
            self.detail_panel.clear()
            return
        function_name = current.data(Qt.UserRole)
        function = self.function_map.get(function_name)
        if function is None:
            self.detail_panel.clear()
            return
        self.detail_panel.display_function(function_name, function)

    def _handle_run_request(self, name: str, params: dict) -> None:
        try:
            result = self.manager.run_function(name, **params)
        except Exception as exc:
            self._log_exception(f"Failed to run {name}", exc)
            QMessageBox.critical(self, "Execution error", str(exc))
            return
        self.append_log(f"Executed {name} with {params!r}")
        self._update_summary()
        self._refresh_functions(keep_selection=name)

    def _handle_queue_request(self, name: str, params: dict) -> None:
        self.manager.functions.append((name, params))
        detail = _function_detail(self.function_map[name])
        label = detail.label if detail else name
        display = f"{label} — {params}"
        self.queue_panel.add_item(display)
        self.append_log(f"Queued {name} with {params!r}")

    def _remove_queue_item(self, row: int) -> None:
        if 0 <= row < len(self.manager.functions):
            removed = self.manager.functions.pop(row)
            self.queue_panel.remove_selected()
            self.append_log(f"Removed {removed[0]} from queue")

    def _clear_queue(self) -> None:
        self.manager.functions.clear()
        self.queue_panel.clear()
        self.append_log("Cleared the queue.")

    def _run_queue(self) -> None:
        if not self.manager.functions:
            self.append_log("Queue is empty.")
            return
        for name, params in list(self.manager.functions):
            try:
                self.manager.run_function(name, **params)
            except Exception as exc:
                self._log_exception(f"Failed to run {name} from queue", exc)
                QMessageBox.critical(self, "Queue execution error", str(exc))
                break
            else:
                self.append_log(f"Queue executed {name} with {params!r}")
        self.manager.functions.clear()
        self.queue_panel.clear()
        self._update_summary()
        self._refresh_functions()

    # Helpers ------------------------------------------------------------------

    def _refresh_functions(self, keep_selection: Optional[str] = None) -> None:
        try:
            self.function_map = self.manager.find_functions()
        except Exception as exc:
            self._log_exception("Failed to enumerate functions", exc)
            QMessageBox.critical(self, "Function discovery error", str(exc))
            self.function_map = {}

        previous_name = keep_selection
        if previous_name is None and self.function_list.currentItem() is not None:
            previous_name = self.function_list.currentItem().data(Qt.UserRole)

        self.function_list.blockSignals(True)
        self.function_list.clear()
        for name, function in self.function_map.items():
            detail = _function_detail(function)
            display = detail.label if detail and detail.label else name
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, name)
            self.function_list.addItem(item)

        self.function_list.blockSignals(False)

        if previous_name:
            for row in range(self.function_list.count()):
                item = self.function_list.item(row)
                if item.data(Qt.UserRole) == previous_name:
                    self.function_list.setCurrentItem(item)
                    break
        elif self.function_list.count():
            self.function_list.setCurrentRow(0)
        else:
            self.detail_panel.clear()

    def _update_summary(self) -> None:
        subjects = self.manager.state.subjects
        if not subjects:
            self.summary_label.setText("No subjects loaded.")
            return
        parts = [f"Subjects: {len(subjects)}"]
        total_trials = sum(len(subj.trials) for subj in subjects)
        parts.append(f"Total trials: {total_trials}")
        class_counts = {
            label
            for subj in subjects
            for label in (t.mapped_label if t.mapped_label is not None else t.raw_label for t in subj.trials)
        }
        parts.append(f"Classes observed: {len(class_counts)}")
        self.summary_label.setText("\n".join(parts))

    def append_log(self, message: str) -> None:
        self.log_view.appendPlainText(f"[{_current_timestamp()}] {message}")

    def _log_exception(self, prefix: str, exc: Exception) -> None:
        self.append_log(f"{prefix}: {exc}")
        self.append_log("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

