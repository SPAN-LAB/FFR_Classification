from __future__ import annotations

import inspect
import io
import json
import math
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QTextCursor
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
    QComboBox,
    QFormLayout,
    QLineEdit,
)

_cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "ffr_gui_cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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
        
        # Set window icon to the logo
        logo_path = Path(__file__).resolve().parent.parent.parent / "spanlab_logo_final.png"
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))

        self.manager = Manager()
        self.function_map: dict[str, Callable] = {}
        self._pipeline_functions: list[dict] = []
        self._selected_index: int | None = None
        self._pending_queue: list[tuple[str, dict]] = []
        self._plot_layouts: list[QVBoxLayout] = []
        self._signals_grid: QGridLayout | None = None
        self._confusion_layout: QVBoxLayout | None = None
        self._roc_layout: QVBoxLayout | None = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[_FunctionWorker] = None
        self._checkpoint_path = Path.cwd() / ".ffr_gui_autosave.pkl"
        self._checkpoint_loaded_for_resume = False

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

        self._load_checkpoint_btn = QPushButton("+  Load Checkpoint")
        self._load_checkpoint_btn.setStyleSheet(_OUTLINED_BTN_STYLE)
        self._load_checkpoint_btn.setFixedHeight(34)
        self._load_checkpoint_btn.clicked.connect(self._load_checkpoint)
        layout.addWidget(self._load_checkpoint_btn)

        self._save_checkpoint_btn = QPushButton("+  Save Checkpoint")
        self._save_checkpoint_btn.setStyleSheet(_OUTLINED_BTN_STYLE)
        self._save_checkpoint_btn.setFixedHeight(34)
        self._save_checkpoint_btn.clicked.connect(self._save_checkpoint_as)
        layout.addWidget(self._save_checkpoint_btn)

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
        self._accuracy_label = QLabel("")
        self._accuracy_label.setAlignment(Qt.AlignCenter)
        self._accuracy_label.setStyleSheet(
            "color: #1a73e8; font-size: 14px; font-weight: bold;"
            " border: none; background: transparent; padding: 4px;"
        )
        self._confusion_layout = panel.layout()
        self._confusion_layout.addWidget(self._accuracy_label)
        self._confusion_layout.addWidget(lbl, stretch=1)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(_PANEL_STYLE)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return scroll

    def _build_roc_panel(self) -> QWidget:
        panel = _titled_panel("ROC Curve")
        lbl = QLabel("(Placeholder)")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            "color: #aaa; font-size: 13px; border: none; background: transparent;"
        )
        self._auc_label = QLabel("")
        self._auc_label.setAlignment(Qt.AlignCenter)
        self._auc_label.setStyleSheet(
            "color: #1a73e8; font-size: 13px;"
            " border: none; background: transparent; padding: 4px;"
        )
        self._roc_layout = panel.layout()
        self._roc_layout.addWidget(self._auc_label)
        self._roc_layout.addWidget(lbl, stretch=1)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(_PANEL_STYLE)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return scroll

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
            " background: white; color: #333333; }"
            " QListWidget::item { padding: 3px 8px; }"
            " QListWidget::item:hover { background: #f0f4ff; }"
        )
        self._subject_list.itemClicked.connect(self._on_subject_clicked)
        layout.addWidget(self._subject_list)
        return panel

    def _build_signals_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(_PANEL_STYLE + "QScrollArea { border: none; }")
        panel = QWidget()
        panel.setStyleSheet("background: white;")
        grid = QGridLayout()
        grid.setSpacing(6)
        grid.setContentsMargins(8, 8, 8, 8)
        panel.setLayout(grid)
        scroll.setWidget(panel)
        self._signals_grid = grid
        self._set_signal_placeholder()
        return scroll

    def _clear_signal_plots(self) -> None:
        if self._signals_grid is None:
            return
        while self._signals_grid.count():
            child = self._signals_grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._plot_layouts = []

    def _add_signal_plot_slot(self, index: int, total: int) -> QVBoxLayout:
        if self._signals_grid is None:
            raise ValueError("Signal plot grid is not initialized.")
        columns = min(4, max(1, math.ceil(math.sqrt(total))))
        row = index // columns
        col = index % columns
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setMinimumHeight(220)
        frame.setStyleSheet(
            "QFrame { border: 1px solid #ddd; border-radius: 4px;"
            " background: #fafafa; }"
        )
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        frame.setLayout(layout)
        self._signals_grid.addWidget(frame, row, col)
        self._plot_layouts.append(layout)
        return layout

    def _set_signal_placeholder(self) -> None:
        self._clear_signal_plots()
        layout = self._add_signal_plot_slot(0, 1)
        lbl = QLabel("Select a subject to view waveforms.")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            "color: #aaa; font-size: 12px; border: none;"
            " background: transparent;"
        )
        layout.addWidget(lbl, stretch=1)

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

        self._add_function_btn = QPushButton("Add Function")
        self._add_function_btn.setStyleSheet(_OUTLINED_BTN_STYLE)
        self._add_function_btn.setFixedSize(160, 40)
        self._add_function_btn.clicked.connect(self._add_function)
        bar.addWidget(self._add_function_btn)

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
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setWindowTitle("Add Function")
        dialog.setMinimumWidth(320)
        dlayout = QVBoxLayout()
        
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Select a function to add:"))
        header_layout.addStretch()
        
        help_link = QLabel("<a href='https://github.com/SPAN-LAB/FFR_Classification'>Help (?)</a>")
        help_link.setOpenExternalLinks(True)
        header_layout.addWidget(help_link)
        
        dlayout.addLayout(header_layout)

        lw = QListWidget()
        for name, func in self.function_map.items():
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

        params = self._default_params_for_function(func_name, func, detail)
        label = detail.label if detail and detail.label else func_name

        self._pipeline_functions.append(
            {"name": func_name, "label": label, "params": params, "detail": detail}
        )
        self._rebuild_cards()
        self._checkpoint_loaded_for_resume = False
        self._autosave_checkpoint()

    def _default_params_for_function(
        self,
        func_name: str,
        func: Callable,
        detail: Optional[FunctionDetail],
    ) -> dict[str, Any]:
        if func_name == "evaluate_model":
            model_name, training_options = self._default_model_config()
            return {
                "model_name": model_name,
                "training_options": training_options,
            }
        if func_name == "train_model":
            model_name, training_options = self._default_model_config()
            return {
                "model_name": model_name,
                "hyperparameters": training_options,
                "output_dirpath": "",
            }

        params: dict[str, Any] = {}
        if not detail:
            return params

        sig = inspect.signature(func)
        param_names = [p for p in sig.parameters.keys() if p != "self"]
        for i, ad in enumerate(detail.argument_details):
            param_name = param_names[i] if i < len(param_names) else f"arg_{i}"
            params[param_name] = "" if isinstance(ad.default_value, Selection) else ad.default_value
        return params

    def _default_model_config(self) -> tuple[str, dict[str, Any]]:
        try:
            from ..models.utils import find_models

            model_names = sorted(find_models().keys())
        except Exception:
            model_names = []

        model_name = "LDA" if "LDA" in model_names else (model_names[0] if model_names else "")
        if model_name == "LDA":
            return model_name, {"solver": "lsqr", "shrinkage": "auto"}
        return model_name, {
            "num_epochs": 20,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.1,
        }

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
        self._checkpoint_loaded_for_resume = False
        self._autosave_checkpoint()

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
        self._checkpoint_loaded_for_resume = False
        self._autosave_checkpoint()

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
        self._checkpoint_loaded_for_resume = False
        self._autosave_checkpoint()

    def _choose_subject_file(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Subject File(s)", str(Path.cwd()), "MAT files (*.mat)"
        )
        if not file_paths:
            return

        try:
            import pymatreader
            raw = pymatreader.read_mat(file_paths[0])
            skip = {"__header__", "__version__", "__globals__", "labels", "time"}
            data_vars = [key for key in raw.keys() if key not in skip]
        except Exception:
            data_vars = ["ffr_nodss"]

        selected_var = "ffr_nodss"
        if len(data_vars) > 1:
            dialog = QDialog(self)
            dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            dialog.setWindowTitle("Select Data Variable")
            dialog.setMinimumWidth(280)
            dlayout = QVBoxLayout()
            dlayout.addWidget(QLabel("Which variable contains the EEG data?"))

            combo = QComboBox()
            combo.setStyleSheet(
                "QComboBox { color: #333; background: white; border: 1px solid #ccc;"
                " border-radius: 4px; padding: 4px 6px; }"
                " QComboBox QAbstractItemView { background: white; color: #333;"
                " selection-background-color: #e8f0fe; selection-color: #111; }"
            )
            for var_name in data_vars:
                combo.addItem(var_name)
            if "ffr_nodss" in data_vars:
                combo.setCurrentIndex(data_vars.index("ffr_nodss"))

            dlayout.addWidget(combo)
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            dlayout.addWidget(buttons)
            dialog.setLayout(dlayout)

            if dialog.exec_() != QDialog.Accepted:
                return
            selected_var = combo.currentText()
        elif data_vars:
            selected_var = data_vars[0]

        try:
            for i, file_path in enumerate(file_paths):
                self.manager.load_subjects(
                    file_path,
                    reset=(i == 0),
                    data_var=selected_var,
                )
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self._update_subjects()
        self._refresh_function_map()
        self._checkpoint_loaded_for_resume = False
        self._autosave_checkpoint()

    def _update_subjects(self) -> None:
        self._subject_list.clear()
        for subj in self.manager.state.subjects:
            self._subject_list.addItem(subj.name)

    def _on_subject_clicked(self, item: QListWidgetItem) -> None:
        subj_name = item.text()
        subject = next((s for s in self.manager.state.subjects if s.name == subj_name), None)
        if not subject:
            return
            
        from ..core import plots
        from ..core import EEGSubject

        mpl.rcParams.update({
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
        })

        self._clear_signal_plots()
        plt.close('all')

        # Group waveform plots by raw tone label so mapped classification labels
        # do not collapse distinct stimuli into one averaged waveform.
        grouped = subject.grouped_trials(key=lambda trial: trial.raw_label)

        try:
            keys = sorted(
                list(grouped.keys()),
                key=lambda value: int(value) if str(value).isdigit() else str(value),
            )
        except Exception:
            keys = list(grouped.keys())

        import warnings
        import seaborn as sns

        sns.set_palette("deep")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if not keys:
                self._set_signal_placeholder()

            for i, group_key in enumerate(keys):
                plot_layout = self._add_signal_plot_slot(i, len(keys))
                    
                trials = grouped[group_key]
                if not trials:
                    plot_layout.addWidget(QLabel(f"No data for Raw Label {group_key}"))
                    continue
                    
                # Create a pseudo-subject with just these trials to average them
                pseudo_subject = EEGSubject(trials=trials)
                pseudo_subject.subaverage(size=len(trials))
                
                if pseudo_subject.trials:
                    avg_trial = pseudo_subject.trials[0]
                    # Inject metadata so plot_single_trial creates a nice title
                    avg_trial.trial_index = "Avg"
                    avg_trial.mapped_label = f"Raw Label {group_key}"
                    
                    try:
                        plots.plot_single_trial(avg_trial)
                        fig = plt.gcf()
                        fig.set_size_inches(4, 3)
                        fig.set_tight_layout(True)
                        canvas = FigureCanvas(fig)
                        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                        canvas.draw()
                        plot_layout.addWidget(canvas, stretch=1)
                        plt.close(fig)
                    except Exception as e:
                        traceback.print_exc(file=sys.__stderr__)
                        plot_layout.addWidget(QLabel(f"Failed to plot Raw Label {group_key}:\n{e}"))
                else:
                    plot_layout.addWidget(QLabel(f"Could not average Raw Label {group_key}"))

        self._accuracy_label.setText("")
        self._auc_label.setText("")
        for layout in (self._confusion_layout, self._roc_layout):
            if layout:
                while layout.count() > 2:
                    child = layout.takeAt(2)
                    if child.widget():
                        child.widget().deleteLater()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            original_close = plt.close
            plt.close = lambda *args, **kwargs: None

            try:
                # Confusion Matrix
                try:
                    plots.plot_confusion_matrix(subject=subject, show_popup=False)
                    fig_cm = plt.gcf()
                    if not fig_cm.axes:
                        self._confusion_layout.addWidget(QLabel("No valid predictions yet."))
                    else:
                        n_classes = len(subject.labels_map)
                        fig_cm.set_size_inches(
                            max(4, n_classes * 0.5),
                            max(4, n_classes * 0.5),
                        )
                        fig_cm.set_tight_layout(True)
                        canvas_cm = FigureCanvas(fig_cm)
                        canvas_cm.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                        canvas_cm.draw()
                        self._confusion_layout.addWidget(canvas_cm, stretch=1)
                        try:
                            from ..core.eeg_trial import EEGTrial as _EEGTrial
                            acc = _EEGTrial.get_accuracy(subject.trials)
                            self._accuracy_label.setText(f"Accuracy: {acc:.2%}")
                        except Exception:
                            self._accuracy_label.setText("")
                    original_close(fig_cm)
                except Exception as e:
                    traceback.print_exc(file=sys.__stderr__)
                    self._confusion_layout.addWidget(QLabel(f"No Confusion Matrix available.\n{e}"))

                # ROC Curve
                try:
                    plots.plot_roc_curve(subject=subject, show_popup=False)
                    fig_roc = plt.gcf()
                    if not fig_roc.axes:
                        self._roc_layout.addWidget(QLabel("No valid predictions yet."))
                    else:
                        n_classes = len(subject.labels_map)
                        fig_roc.set_size_inches(
                            max(10, n_classes * 0.6),
                            max(5, n_classes * 0.35),
                        )
                        fig_roc.set_tight_layout(True)
                        canvas_roc = FigureCanvas(fig_roc)
                        canvas_roc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                        canvas_roc.draw()
                        self._roc_layout.addWidget(canvas_roc, stretch=1)
                        try:
                            from sklearn.metrics import roc_auc_score
                            classes = sorted(
                                subject.labels_map.keys(),
                                key=lambda value: int(value) if str(value).isdigit() else str(value),
                            )
                            y_true, y_scores = [], []
                            for trial in subject.trials:
                                if trial.prediction_distribution:
                                    y_true.append(trial.label)
                                    y_scores.append([
                                        trial.prediction_distribution.get(label, 0)
                                        for label in classes
                                    ])
                            if y_true:
                                y_true_bin = [
                                    [1 if true == label else 0 for label in classes]
                                    for true in y_true
                                ]
                                auc_scores = roc_auc_score(y_true_bin, y_scores, average=None)
                                auc_text = "  ".join(
                                    f"T{label}: {auc:.3f}"
                                    for label, auc in zip(classes, auc_scores)
                                )
                                self._auc_label.setText(f"AUC: {auc_text}")
                        except Exception:
                            self._auc_label.setText("")
                    original_close(fig_roc)
                except Exception as e:
                    traceback.print_exc(file=sys.__stderr__)
                    self._roc_layout.addWidget(QLabel(f"No ROC Curve available.\n{e}"))
            finally:
                plt.close = original_close

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
            self._checkpoint_loaded_for_resume = False
            self._autosave_checkpoint()

        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Could not load pipeline:\n{exc}")

    def _checkpoint_payload_functions(self) -> list[dict]:
        payload = []
        for func in self._pipeline_functions:
            payload.append({
                "name": func["name"],
                "label": func["label"],
                "params": func["params"],
            })
        return payload

    def _restore_checkpoint_functions(self, saved_functions: list[dict]) -> None:
        self._refresh_function_map()
        restored = []
        for item in saved_functions:
            name = item.get("name")
            func = self.function_map.get(name)
            detail = _function_detail(func) if func else None
            restored.append({
                "name": name,
                "label": item.get("label", name),
                "params": item.get("params", {}),
                "detail": detail,
            })
        self._pipeline_functions = restored
        self._selected_index = None
        self._param_editor.clear_and_hide()
        self._rebuild_cards()

    def _autosave_checkpoint(self) -> None:
        try:
            self.manager.save_checkpoint(
                self._checkpoint_path,
                pipeline_functions=self._checkpoint_payload_functions(),
                pending_queue=list(self._pending_queue),
                completed_steps=getattr(self, "_completed_steps", 0),
                log_text=self._log_text,
            )
        except Exception as exc:
            self._append_log(f"Checkpoint save failed: {exc}\n")

    def _save_checkpoint_as(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Checkpoint", str(Path.cwd() / "ffr_gui_checkpoint.pkl"), "Pickle files (*.pkl)"
        )
        if not file_path:
            return
        try:
            self.manager.save_checkpoint(
                file_path,
                pipeline_functions=self._checkpoint_payload_functions(),
                pending_queue=list(self._pending_queue),
                completed_steps=getattr(self, "_completed_steps", 0),
                log_text=self._log_text,
            )
            QMessageBox.information(self, "Success", "Checkpoint saved successfully.")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Could not save checkpoint:\n{exc}")

    def _load_checkpoint(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Checkpoint", str(Path.cwd()), "Pickle files (*.pkl)"
        )
        if not file_path:
            return
        try:
            payload = self.manager.load_checkpoint(file_path)
            self._restore_checkpoint_functions(payload.get("pipeline_functions", []))
            self._pending_queue = payload.get("pending_queue", [])
            self._completed_steps = int(payload.get("completed_steps", 0))
            self._log_text = payload.get("log_text", "")
            self._update_subjects()
            self._refresh_function_map()
            self._checkpoint_loaded_for_resume = bool(self._pending_queue)
            self._status_label.setText("Checkpoint loaded. Click to view log.")
            self._status_label.show()
            QMessageBox.information(self, "Success", "Checkpoint loaded successfully.")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Could not load checkpoint:\n{exc}")

    # ── pipeline execution ───────────────────────────────────────────────────

    def _run_pipeline(self) -> None:
        if not self._pipeline_functions:
            QMessageBox.information(
                self, "Empty Pipeline", "Add functions to the pipeline first."
            )
            return

        if self._checkpoint_loaded_for_resume and self._pending_queue:
            remaining_steps = len(self._pending_queue)
            self._total_steps = self._completed_steps + remaining_steps
            self._append_log(f"--- Resuming {remaining_steps} pending step(s) ---\n")
        else:
            self.manager.reset_to_initial()
            self._refresh_function_map()
            self._pending_queue = [
                (f["name"], dict(f["params"])) for f in self._pipeline_functions
            ]
            self._total_steps = len(self._pending_queue)
            self._completed_steps = 0
            self._log_text = ""
            if self._log_dialog_text is not None:
                self._log_dialog_text.setPlainText("")

        self._checkpoint_loaded_for_resume = False
        self._progress_bar.setMaximum(self._total_steps)
        self._progress_bar.setValue(self._completed_steps)
        self._progress_bar.setFormat(f"{self._completed_steps}/{self._total_steps}")
        self._progress_bar.show()
        self._status_label.show()
        self._autosave_checkpoint()
        self._start_next_queued()

    def _start_next_queued(self) -> None:
        if not self._pending_queue:
            self._set_running(False)
            self._update_subjects()
            self._refresh_selected_subject_plots()
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
            self._set_running(False)
            self._autosave_checkpoint()
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
        self._autosave_checkpoint()
        if self._pending_queue:
            self._start_next_queued()
        else:
            self._set_running(False)
            self._update_subjects()
            self._refresh_selected_subject_plots()
            self._status_label.setText("Pipeline finished. Click to view log.")
            QMessageBox.information(
                self, "Complete", "Pipeline execution finished."
            )

    def _on_run_failed(self, message: str, _tb: str) -> None:
        self._thread = None
        self._worker = None
        self._append_log(f"ERROR: {message}\n")
        self._set_running(False)
        self._autosave_checkpoint()
        self._checkpoint_loaded_for_resume = bool(self._pending_queue)
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
            " background: white; color: #222; border: 1px solid #ccc; border-radius: 4px;"
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

    def _refresh_selected_subject_plots(self) -> None:
        current = self._subject_list.currentItem()
        if current is not None:
            self._on_subject_clicked(current)

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._load_subjects_btn.setEnabled(not running)
        self._load_subject_file_btn.setEnabled(not running)
        self._load_checkpoint_btn.setEnabled(not running)
        self._load_pipe_btn.setEnabled(not running)
        self._save_pipe_btn.setEnabled(not running)
        self._save_checkpoint_btn.setEnabled(not running)
        self._add_function_btn.setEnabled(not running)
        self._scroll.setEnabled(not running)
        self._param_editor.setEnabled(not running)


def main() -> None:
    app = QApplication(sys.argv)
    # app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
