from __future__ import annotations

import inspect
import io
import json
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .manager import Manager
from .widgets import FunctionCardWidget, ParameterEditorWidget
from ..core.utils.function_detail import FunctionDetail, Selection


def _function_detail(fn: Callable) -> Optional[FunctionDetail]:
    func = getattr(fn, "__func__", fn)
    return getattr(func, "detail", None)


class _FdCapture:
    """Tee sys.stdout and sys.stderr to both the real terminal and
    an in-memory buffer. Safer on Windows than fd-level redirection."""

    def __init__(
        self, on_chunk: Optional[Callable[[str], None]] = None
    ) -> None:
        self._buf = io.StringIO()
        self._lock = threading.Lock()
        self._on_chunk = on_chunk
        self._old_stdout: Optional[io.TextIOWrapper] = None
        self._old_stderr: Optional[io.TextIOWrapper] = None

    class _Tee:
        def __init__(self, primary, secondary, lock, on_chunk):
            self.primary = primary
            self.secondary = secondary
            self.lock = lock
            self.on_chunk = on_chunk

        def write(self, data):
            if not data:
                return
            try:
                self.primary.write(data)
            except Exception:
                pass
            with self.lock:
                self.secondary.write(data)
            if self.on_chunk:
                try:
                    self.on_chunk(data)
                except Exception:
                    pass

        def flush(self):
            try:
                self.primary.flush()
            except Exception:
                pass
            with self.lock:
                self.secondary.flush()
        
        def isatty(self):
            return getattr(self.primary, "isatty", lambda: False)()

    def start(self) -> None:
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = self._Tee(self._old_stdout, self._buf, self._lock, self._on_chunk)
        sys.stderr = self._Tee(self._old_stderr, self._buf, self._lock, self._on_chunk)

    def stop(self) -> str:
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
            self._old_stdout = None
        if self._old_stderr is not None:
            sys.stderr = self._old_stderr
            self._old_stderr = None
        with self._lock:
            return self._buf.getvalue()


class _FunctionWorker(QObject):
    finished = pyqtSignal()
    failed = pyqtSignal(str, str)
    output_chunk = pyqtSignal(str)

    def __init__(self, fn: Callable, kwargs: dict) -> None:
        super().__init__()
        self._fn = fn
        self._kwargs = kwargs

    def run(self) -> None:
        cap = _FdCapture(on_chunk=lambda text: self.output_chunk.emit(text))
        cap.start()
        try:
            self._fn(**self._kwargs)
        except Exception as exc:
            cap.stop()
            self.failed.emit(str(exc), traceback.format_exc())
            return
        cap.stop()
        self.finished.emit()


_PANEL_STYLE = """
    background: white;
    border: 1px solid #d8d8d8;
    border-radius: 6px;
"""

_RUN_BTN_STYLE = """
    QPushButton {
        background: #4285f4; color: white; border: none;
        border-radius: 6px; padding: 10px 24px;
        font-size: 13px; font-weight: bold;
    }
    QPushButton:hover { background: #3367d6; }
    QPushButton:disabled { background: #ccc; }
"""

_OUTLINED_BTN_STYLE = """
    QPushButton {
        background: white; color: #333; border: 1px solid #ccc;
        border-radius: 6px; padding: 8px 18px;
        font-size: 13px; font-weight: bold;
    }
    QPushButton:hover { background: #f5f5f5; border-color: #aaa; }
    QPushButton:disabled { background: #eee; color: #999; }
"""


def _titled_panel(title: str) -> QWidget:
    panel = QWidget()
    panel.setStyleSheet(_PANEL_STYLE)
    layout = QVBoxLayout()
    layout.setContentsMargins(10, 10, 10, 10)
    lbl = QLabel(title)
    lbl.setStyleSheet(
        "font-weight: bold; font-size: 13px; color: #333;"
        " border: none; background: transparent;"
    )
    layout.addWidget(lbl)
    panel.setLayout(layout)
    return panel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FFR Pipeline Tool")
        self.resize(1400, 900)

        self.manager = Manager()
        self.function_map: dict[str, Callable] = {}
        self._pipeline_functions: list[dict[str, Any]] = []
        self._selected_index: Optional[int] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[_FunctionWorker] = None
        self._pending_queue: list[tuple[str, dict]] = []

        self._build_ui()
        self._refresh_function_map()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        central.setStyleSheet("background: #eaeaea;")
        self.setCentralWidget(central)
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        central.setLayout(root)

        root.addWidget(self._build_header())
        root.addWidget(self._build_content(), stretch=1)
        root.addWidget(self._build_bottom_bar())

    # ── header ───────────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(70)
        header.setStyleSheet(
            "background: white; border-bottom: 1px solid #d0d0d0;"
        )
        layout = QHBoxLayout()
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(10)
        header.setLayout(layout)

        logo_label = QLabel()
        logo_path = (
            Path(__file__).resolve().parent.parent.parent / "spanlab_logo_final.png"
        )
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            logo_label.setPixmap(
                pix.scaled(58, 58, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            logo_label.setText("SPANLAB")
            logo_label.setStyleSheet(
                "font-size: 20px; font-weight: bold; color: #8B0000;"
            )
        layout.addWidget(logo_label)
        layout.addStretch()

        self._load_pipe_btn = QPushButton("+  Load Pipeline")
        self._load_pipe_btn.setStyleSheet(_OUTLINED_BTN_STYLE)
        self._load_pipe_btn.setFixedHeight(34)
        self._load_pipe_btn.clicked.connect(self._load_pipeline)
        layout.addWidget(self._load_pipe_btn)

        self._save_pipe_btn = QPushButton("+  Save Pipeline")
        self._save_pipe_btn.setStyleSheet(_OUTLINED_BTN_STYLE)
        self._save_pipe_btn.setFixedHeight(34)
        self._save_pipe_btn.clicked.connect(self._save_pipeline)
        layout.addWidget(self._save_pipe_btn)

        self._load_subject_file_btn = QPushButton("+  Load Subject File")
        self._load_subject_file_btn.setStyleSheet(_OUTLINED_BTN_STYLE)
        self._load_subject_file_btn.setFixedHeight(34)
        self._load_subject_file_btn.clicked.connect(self._choose_subject_file)
        layout.addWidget(self._load_subject_file_btn)

        self._load_subjects_btn = QPushButton("+  Load Subject Folder")
        self._load_subjects_btn.setStyleSheet(_OUTLINED_BTN_STYLE)
        self._load_subjects_btn.setFixedHeight(34)
        self._load_subjects_btn.clicked.connect(self._choose_subjects)
        layout.addWidget(self._load_subjects_btn)

        return header

    # ── main content with splitters ──────────────────────────────────────────

    def _build_content(self) -> QSplitter:
        # Top level: left panel | right area
        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.setStyleSheet(
            "QSplitter { background: #eaeaea; }"
            " QSplitter::handle { background: #d0d0d0; width: 3px; height: 3px; }"
        )
        self._main_splitter.setHandleWidth(4)
        self._main_splitter.setContentsMargins(8, 8, 8, 0)

        self._main_splitter.addWidget(self._build_left_panel())

        # Right area: top row / bottom row
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setHandleWidth(4)
        right_splitter.setStyleSheet(
            "QSplitter::handle { background: #d0d0d0; }"
        )

        # Top right: confusion matrix | ROC curve
        top_right = QSplitter(Qt.Horizontal)
        top_right.setHandleWidth(4)
        top_right.setStyleSheet(
            "QSplitter::handle { background: #d0d0d0; }"
        )
        top_right.addWidget(self._build_confusion_panel())
        top_right.addWidget(self._build_roc_panel())
        top_right.setSizes([500, 500])

        # Bottom right: subjects | signal plots
        bottom_right = QSplitter(Qt.Horizontal)
        bottom_right.setHandleWidth(4)
        bottom_right.setStyleSheet(
            "QSplitter::handle { background: #d0d0d0; }"
        )
        bottom_right.addWidget(self._build_subjects_panel())
        bottom_right.addWidget(self._build_signals_panel())
        bottom_right.setSizes([180, 700])

        right_splitter.addWidget(top_right)
        right_splitter.addWidget(bottom_right)
        right_splitter.setSizes([420, 380])

        self._main_splitter.addWidget(right_splitter)
        self._main_splitter.setSizes([300, 1050])

        return self._main_splitter

    # ── left panel (functions + editor) ──────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(260)
        panel.setStyleSheet("background: #f2f2f2; border-radius: 6px;")
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        panel.setLayout(layout)

        self._cards_layout = QVBoxLayout()
        self._cards_layout.setSpacing(4)
        self._cards_layout.setContentsMargins(2, 2, 2, 2)
        self._cards_layout.addStretch()

        cards_widget = QWidget()
        cards_widget.setStyleSheet("background: transparent;")
        cards_widget.setLayout(self._cards_layout)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(cards_widget)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        layout.addWidget(self._scroll, stretch=1)

        self._param_editor = ParameterEditorWidget(self.manager)
        self._param_editor.apply_clicked.connect(self._on_apply_params)
        layout.addWidget(self._param_editor)

        return panel

    # ── right-side panels ────────────────────────────────────────────────────

    def _build_confusion_panel(self) -> QWidget:
        panel = _titled_panel("Confusion Matrix")
        lbl = QLabel("(Placeholder)")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            "color: #aaa; font-size: 13px; border: none; background: transparent;"
        )
        panel.layout().addWidget(lbl, stretch=1)
        return panel

    def _build_roc_panel(self) -> QWidget:
        panel = _titled_panel("ROC Curve")
        lbl = QLabel("(Placeholder)")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            "color: #aaa; font-size: 13px; border: none; background: transparent;"
        )
        panel.layout().addWidget(lbl, stretch=1)
        return panel

    def _build_subjects_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(_PANEL_STYLE)
        panel.setMinimumWidth(140)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        panel.setLayout(layout)

        self._subject_list = QListWidget()
        self._subject_list.setStyleSheet(
            "QListWidget { border: none; font-size: 12px;"
            " background: white; }"
            " QListWidget::item { padding: 3px 8px; }"
            " QListWidget::item:hover { background: #f0f4ff; }"
        )
        layout.addWidget(self._subject_list)
        return panel

    def _build_signals_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(_PANEL_STYLE)
        grid = QGridLayout()
        grid.setSpacing(6)
        grid.setContentsMargins(8, 8, 8, 8)

        for r in range(2):
            for c in range(2):
                frame = QFrame()
                frame.setFrameShape(QFrame.StyledPanel)
                frame.setStyleSheet(
                    "QFrame { border: 1px solid #ddd; border-radius: 4px;"
                    " background: #fafafa; }"
                )
                fl = QVBoxLayout()
                fl.setContentsMargins(4, 4, 4, 4)
                lbl = QLabel(f"Signal Plot {r * 2 + c + 1}")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet(
                    "color: #aaa; font-size: 12px; border: none;"
                    " background: transparent;"
                )
                fl.addWidget(lbl, stretch=1)
                frame.setLayout(fl)
                grid.addWidget(frame, r, c)

        panel.setLayout(grid)
        return panel

    # ── bottom bar ───────────────────────────────────────────────────────────

    def _build_bottom_bar(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet("background: #eaeaea;")
        outer = QVBoxLayout()
        outer.setContentsMargins(8, 0, 8, 8)
        outer.setSpacing(0)
        container.setLayout(outer)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: rgba(0,0,0,0.08);")
        outer.addWidget(sep)

        bar = QHBoxLayout()
        bar.setContentsMargins(4, 8, 4, 0)

        add_btn = QPushButton("Add Function")
        add_btn.setStyleSheet(_OUTLINED_BTN_STYLE)
        add_btn.setFixedSize(160, 40)
        add_btn.clicked.connect(self._add_function)
        bar.addWidget(add_btn)

        bar.addSpacing(16)

        self._run_btn = QPushButton("Run Functions")
        self._run_btn.setStyleSheet(_RUN_BTN_STYLE)
        self._run_btn.setFixedSize(160, 40)
        self._run_btn.clicked.connect(self._run_pipeline)
        bar.addWidget(self._run_btn)

        bar.addSpacing(12)

        self._status_label = QPushButton("")
        self._status_label.setFlat(True)
        self._status_label.setStyleSheet(
            "QPushButton { background: transparent; color: #555; border: none;"
            " font-size: 12px; text-align: left; padding: 0; }"
            " QPushButton:hover { color: #1a73e8; text-decoration: underline; }"
        )
        self._status_label.setCursor(Qt.PointingHandCursor)
        self._status_label.clicked.connect(self._show_log)
        self._status_label.hide()
        bar.addWidget(self._status_label)

        bar.addStretch()

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(260)
        self._progress_bar.setFixedHeight(18)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #ccc; border-radius: 4px;"
            " background: #e0e0e0; text-align: center; font-size: 11px; color: #333; }"
            " QProgressBar::chunk { background: #4285f4; border-radius: 3px; }"
        )
        self._progress_bar.setValue(0)
        self._progress_bar.hide()
        bar.addWidget(self._progress_bar)

        outer.addLayout(bar)

        self._log_text: str = ""
        self._log_dialog_text: Optional[QPlainTextEdit] = None

        return container

    # ── function map ─────────────────────────────────────────────────────────

    def _refresh_function_map(self) -> None:
        try:
            self.function_map = self.manager.find_functions()
        except Exception:
            self.function_map = {}

    # ── add / edit / apply ───────────────────────────────────────────────────

    def _add_function(self) -> None:
        self._refresh_function_map()
        if not self.function_map:
            QMessageBox.information(
                self, "No Functions", "No pipeline functions available."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Add Function")
        dialog.setMinimumWidth(320)
        dlayout = QVBoxLayout()
        dlayout.addWidget(QLabel("Select a function to add:"))

        already_added = {f["name"] for f in self._pipeline_functions}
        lw = QListWidget()
        for name, func in self.function_map.items():
            if name in already_added:
                continue
            det = _function_detail(func)
            display = det.label if det and det.label else name
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, name)
            lw.addItem(item)
        if lw.count() == 0:
            QMessageBox.information(
                self, "No Functions",
                "All available functions have already been added."
            )
            return
        dlayout.addWidget(lw)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        dlayout.addWidget(btns)
        dialog.setLayout(dlayout)

        if dialog.exec_() != QDialog.Accepted or lw.currentItem() is None:
            return

        func_name = lw.currentItem().data(Qt.UserRole)
        func = self.function_map[func_name]
        detail = _function_detail(func)

        params: dict[str, Any] = {}
        if detail:
            # Extract parameter names from function signature, skip 'self'
            sig = inspect.signature(func)
            param_names = [p for p in sig.parameters.keys() if p != 'self']
            
            for i, ad in enumerate(detail.argument_details):
                # Get the parameter name by position
                param_name = param_names[i] if i < len(param_names) else f"arg_{i}"
                
                if isinstance(ad.default_value, Selection):
                    # For Selection types, use empty string as default (lazy-load options later)
                    params[param_name] = ""
                else:
                    params[param_name] = ad.default_value
            label = detail.label
        else:
            label = func_name

        self._pipeline_functions.append(
            {"name": func_name, "label": label, "params": params, "detail": detail}
        )
        self._rebuild_cards()

    def _rebuild_cards(self) -> None:
        while self._cards_layout.count() > 1:
            item = self._cards_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        for i, entry in enumerate(self._pipeline_functions):
            card = FunctionCardWidget(
                i, entry["label"], entry["params"], entry["detail"]
            )
            card.edit_clicked.connect(self._on_edit_card)
            card.delete_clicked.connect(self._on_delete_card)
            card.set_selected(i == self._selected_index)
            self._cards_layout.insertWidget(i, card)

    def _on_delete_card(self, index: int) -> None:
        self._pipeline_functions.pop(index)
        if self._selected_index == index:
            self._selected_index = None
            self._param_editor.clear_and_hide()
        elif self._selected_index is not None and self._selected_index > index:
            self._selected_index -= 1
        self._rebuild_cards()

    def _on_edit_card(self, index: int) -> None:
        self._selected_index = index
        entry = self._pipeline_functions[index]

        for i in range(self._cards_layout.count()):
            w = self._cards_layout.itemAt(i).widget()
            if isinstance(w, FunctionCardWidget):
                w.set_selected(w._index == index)

        if entry["detail"]:
            self._param_editor.display(entry["detail"], entry["params"], entry["name"])

    def _on_apply_params(self, params: dict) -> None:
        if self._selected_index is None:
            return
        if self._selected_index >= len(self._pipeline_functions):
            return

        self._pipeline_functions[self._selected_index]["params"] = params
        self._rebuild_cards()
        self._param_editor.clear_and_hide()
        self._selected_index = None

    # ── subjects ─────────────────────────────────────────────────────────────

    def _choose_subjects(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Subject Folder", str(Path.cwd())
        )
        if not folder:
            return
        try:
            self.manager.load_subjects(folder)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self._update_subjects()
        self._refresh_function_map()

    def _choose_subject_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Subject File", str(Path.cwd()), "MAT files (*.mat)"
        )
        if not file_path:
            return
        try:
            self.manager.load_subjects(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self._update_subjects()
        self._refresh_function_map()

    def _update_subjects(self) -> None:
        self._subject_list.clear()
        for subj in self.manager.state.subjects:
            self._subject_list.addItem(subj.name)

    # ── pipeline saving / loading ────────────────────────────────────────────

    def _save_pipeline(self) -> None:
        if not self._pipeline_functions:
            QMessageBox.information(
                self, "Empty Pipeline", "No functions to save."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Pipeline", str(Path.cwd()), "JSON files (*.json)"
        )
        if not file_path:
            return

        pipeline_data = []
        for func in self._pipeline_functions:
            # We don't save the 'detail' object because it might not be JSON serializable
            # We'll re-fetch it when loading
            pipeline_data.append({
                "name": func["name"],
                "label": func["label"],
                "params": func["params"]
            })

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(pipeline_data, f, indent=4)
            QMessageBox.information(self, "Success", "Pipeline saved successfully.")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Could not save pipeline:\n{exc}")

    def _load_pipeline(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Pipeline", str(Path.cwd()), "JSON files (*.json)"
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                pipeline_data = json.load(f)

            if not isinstance(pipeline_data, list):
                raise ValueError("Invalid pipeline format (expected a list).")

            self._refresh_function_map()

            new_functions = []
            for item in pipeline_data:
                name = item.get("name")
                if not name:
                    continue
                func = self.function_map.get(name)
                # Even if func is missing (maybe changed version), we load it
                detail = _function_detail(func) if func else None
                new_functions.append({
                    "name": name,
                    "label": item.get("label", name),
                    "params": item.get("params", {}),
                    "detail": detail
                })

            self._pipeline_functions = new_functions
            self._selected_index = None
            self._param_editor.clear_and_hide()
            self._rebuild_cards()

        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Could not load pipeline:\n{exc}")

    # ── pipeline execution ───────────────────────────────────────────────────

    def _run_pipeline(self) -> None:
        if not self._pipeline_functions:
            QMessageBox.information(
                self, "Empty Pipeline", "Add functions to the pipeline first."
            )
            return
        self._pending_queue = [
            (f["name"], dict(f["params"])) for f in self._pipeline_functions
        ]
        self._total_steps = len(self._pending_queue)
        self._completed_steps = 0
        self._log_text = ""
        if self._log_dialog_text is not None:
            self._log_dialog_text.setPlainText("")
        self._progress_bar.setMaximum(self._total_steps)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat(f"0/{self._total_steps}")
        self._progress_bar.show()
        self._status_label.show()
        self._start_next_queued()

    def _start_next_queued(self) -> None:
        if not self._pending_queue:
            self._set_running(False)
            self._update_subjects()
            self._status_label.setText("Pipeline finished. Click to view log.")
            QMessageBox.information(
                self, "Complete", "Pipeline execution finished."
            )
            return

        name, params = self._pending_queue.pop(0)
        func = self.function_map.get(name)
        if func is None:
            self._append_log(f"Error: Unknown function: {name}\n")
            QMessageBox.critical(self, "Error", f"Unknown function: {name}")
            self._pending_queue.clear()
            self._set_running(False)
            return

        detail = _function_detail(func)
        display_name = detail.label if detail and detail.label else name
        subjects = self.manager.state.subjects
        if not subjects:
            subject_suffix = ""
        elif len(subjects) <= 3:
            subject_suffix = f" on {', '.join(s.name for s in subjects)}"
        else:
            subject_suffix = f" on {len(subjects)} subjects"
        self._status_label.setText(f"Evaluating {display_name}{subject_suffix}")
        self._append_log(f"--- {display_name}{subject_suffix} ---\n")

        # Ensure folding if evaluating model and no folds exist
        if name == "evaluate_model":
            for s in subjects:
                if s.folds is None:
                    self._append_log(f"Auto-folding subject {s.name} (5 folds)...\n")
                    s.fold(5)

        self._set_running(True)
        thread = QThread(self)
        worker = _FunctionWorker(func, params)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.output_chunk.connect(self._append_log)
        worker.finished.connect(self._on_run_finished)
        worker.failed.connect(self._on_run_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_run_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._completed_steps += 1
        self._progress_bar.setValue(self._completed_steps)
        self._progress_bar.setFormat(
            f"{self._completed_steps}/{self._total_steps}"
        )
        self._refresh_function_map()
        if self._pending_queue:
            self._start_next_queued()
        else:
            self._set_running(False)
            self._update_subjects()
            self._status_label.setText("Pipeline finished. Click to view log.")
            QMessageBox.information(
                self, "Complete", "Pipeline execution finished."
            )

    def _on_run_failed(self, message: str, _tb: str) -> None:
        self._thread = None
        self._worker = None
        self._append_log(f"ERROR: {message}\n")
        self._pending_queue.clear()
        self._set_running(False)
        self._status_label.setText("Pipeline failed. Click to view log.")
        QMessageBox.critical(self, "Execution Error", message)

    def _append_log(self, text: str) -> None:
        self._log_text += text
        if self._log_dialog_text is not None:
            self._log_dialog_text.moveCursor(QTextCursor.End)
            self._log_dialog_text.insertPlainText(text)
            self._log_dialog_text.moveCursor(QTextCursor.End)

    def _show_log(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Pipeline Log")
        dialog.setMinimumSize(480, 320)
        dialog.setModal(False)
        layout = QVBoxLayout()
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(self._log_text)
        text.setStyleSheet(
            "QPlainTextEdit { font-family: monospace; font-size: 12px;"
            " background: white; border: 1px solid #ccc; border-radius: 4px;"
            " padding: 8px; }"
        )
        text.moveCursor(QTextCursor.End)
        layout.addWidget(text)
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(_OUTLINED_BTN_STYLE)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)
        dialog.setLayout(layout)
        self._log_dialog_text = text
        dialog.finished.connect(lambda _: self._clear_log_dialog_ref())
        dialog.show()

    def _clear_log_dialog_ref(self) -> None:
        self._log_dialog_text = None

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._load_subjects_btn.setEnabled(not running)
        self._load_subject_file_btn.setEnabled(not running)


def main() -> None:
    app = QApplication(sys.argv)
    # app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()