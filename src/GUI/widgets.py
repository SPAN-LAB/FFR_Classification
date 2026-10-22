from __future__ import annotations

import inspect
import json
from typing import Any, Optional

from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .manager import Manager
from ..core.utils.function_detail import ArgumentDetail, FunctionDetail, PathKind, Selection
from ..models.utils import find_models, get_model_import_errors


def _is_dict_type(typ) -> bool:
    if typ is dict:
        return True
    return getattr(typ, "__origin__", None) is dict


class DictEditorWidget(QWidget):
    def __init__(self, initial_dict: dict = None, parent=None):
        super().__init__(parent)
        self.rows = []
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(4)
        self.layout.addLayout(self.rows_layout)
        
        self.add_btn = QPushButton("+ Add Parameter")
        self.add_btn.setStyleSheet(
            "QPushButton { background: #e0e0e0; border: 1px solid #ccc; border-radius: 4px; padding: 2px 6px; font-size: 11px; }"
            "QPushButton:hover { background: #d0d0d0; }"
        )
        self.add_btn.clicked.connect(lambda: self.add_row("", ""))
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.add_btn)
        self.layout.addLayout(btn_layout)
        
        self.setLayout(self.layout)
        
        if initial_dict:
            for k, v in initial_dict.items():
                self.add_row(str(k), self._val_to_str(v))
                
    def _val_to_str(self, val):
        if isinstance(val, (dict, list)):
            return json.dumps(val)
        return str(val)
        
    def add_row(self, key: str, val: str):
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        
        key_edit = QLineEdit(key)
        key_edit.setPlaceholderText("Key")
        key_edit.setStyleSheet("border: 1px solid #ccc; border-radius: 4px; padding: 2px 4px; background: white;")
        
        val_edit = QLineEdit(val)
        val_edit.setPlaceholderText("Value")
        val_edit.setStyleSheet("border: 1px solid #ccc; border-radius: 4px; padding: 2px 4px; background: white;")
        
        del_btn = QPushButton("X")
        del_btn.setFixedSize(20, 20)
        del_btn.setStyleSheet(
            "QPushButton { background: #ffcdd2; color: #c62828; border: 1px solid #ef9a9a; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #ef9a9a; }"
        )
        del_btn.clicked.connect(lambda: self.remove_row(row_widget))
        
        row_layout.addWidget(key_edit)
        row_layout.addWidget(QLabel(":"))
        row_layout.addWidget(val_edit)
        row_layout.addWidget(del_btn)
        
        row_widget.setLayout(row_layout)
        self.rows_layout.addWidget(row_widget)
        self.rows.append((row_widget, key_edit, val_edit))
        
    def remove_row(self, row_widget):
        for i, (rw, k, v) in enumerate(self.rows):
            if rw == row_widget:
                self.rows.pop(i)
                self.rows_layout.removeWidget(rw)
                rw.deleteLater()
                break

    def get_value(self) -> dict:
        res = {}
        for rw, k_edit, v_edit in self.rows:
            key = k_edit.text().strip()
            if not key:
                continue
            val_str = v_edit.text().strip()
            
            # Auto parse
            try:
                if "." in val_str:
                    val = float(val_str)
                else:
                    val = int(val_str)
            except ValueError:
                # Try json parse (for booleans, lists, dicts)
                try:
                    val = json.loads(val_str)
                except json.JSONDecodeError:
                    val = val_str
            res[key] = val
        return res


class FunctionCardWidget(QFrame):
    edit_clicked = pyqtSignal(int)
    delete_clicked = pyqtSignal(int)
    drag_started = pyqtSignal(int, QPoint)
    drag_moved = pyqtSignal(QPoint)
    drag_finished = pyqtSignal(int)
    drag_cancelled = pyqtSignal(int)

    def __init__(
        self,
        index: int,
        label: str,
        params: dict,
        detail: Optional[FunctionDetail],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._index = index
        self._label_text = label
        self._params = dict(params)
        self._detail = detail
        self._press_pos: Optional[QPoint] = None
        self._dragging = False
        self._drag_enabled = True
        self._drag_saved_min_height: Optional[int] = None
        self._drag_saved_max_height: Optional[int] = None

        self.setFrameShape(QFrame.Box)
        self.setLineWidth(1)
        self._apply_style(selected=False)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._description = detail.description if detail and detail.description else None
        self._tooltip_timer = QTimer(self)
        self._tooltip_timer.setSingleShot(True)
        self._tooltip_timer.setInterval(2000)
        self._tooltip_timer.timeout.connect(self._show_delayed_tooltip)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self._title = QLabel(f"<b>{label}</b>")
        self._title.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        header.addWidget(self._title)
        header.addStretch()

        self._edit_btn = QPushButton("Edit")
        self._edit_btn.setFixedSize(50, 24)
        self._edit_btn.setStyleSheet(
            "QPushButton { background: #e0e0e0; color: #333; border: 1px solid #ccc;"
            " border-radius: 4px; font-size: 11px; }"
            " QPushButton:hover { background: #d0d0d0; }"
        )
        self._edit_btn.clicked.connect(lambda: self.edit_clicked.emit(self._index))
        header.addWidget(self._edit_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setFixedSize(50, 24)
        self._delete_btn.setStyleSheet(
            "QPushButton { background: #e0e0e0; color: #c62828; border: 1px solid #ccc;"
            " border-radius: 4px; font-size: 11px; }"
            " QPushButton:hover { background: #ffcdd2; }"
        )
        self._delete_btn.clicked.connect(lambda: self.delete_clicked.emit(self._index))
        header.addWidget(self._delete_btn)
        layout.addLayout(header)

        self._summary_label = QLabel(self._make_summary())
        self._summary_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._summary_label.setStyleSheet("color: #555; font-size: 11px;")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self.setLayout(layout)

    def _make_summary(self) -> str:
        if self._label_text == "Extract Features":
            return self._extract_features_summary()

        if self._label_text == "Infer on Model":
            options = self._params.get("training_options")
            batch_size = options.get("batch_size") if isinstance(options, dict) else None
            if isinstance(batch_size, int) and batch_size > 0:
                return f"Batch Size: {batch_size}"
            return ""

        if "model_name" in self._params:
            parts = [f"Model: {self._params.get('model_name', '')}"]
            options = self._params.get("training_options", self._params.get("hyperparameters"))
            if isinstance(options, dict):
                parts.append(
                    "Options: "
                    + ", ".join(f"{k}: {v}" for k, v in options.items())
                )
            held_out = self._params.get("only_held_out")
            if isinstance(held_out, list) and held_out:
                parts.append("Hold out subjects: " + ", ".join(str(name) for name in held_out))
            save_dir = self._params.get("model_save_dir", self._params.get("output_dirpath"))
            if isinstance(save_dir, str) and save_dir.strip():
                label = "Save Directory"
                if self._detail:
                    for ad in self._detail.argument_details:
                        if ad.type is PathKind.directory:
                            label = ad.label
                            break
                parts.append(f"{label}: {save_dir.strip()}")
            return "\n".join(parts)

        if not self._detail or not self._detail.argument_details:
            return ""
        parts = []
        param_names = list(self._params.keys())
        
        for i, ad in enumerate(self._detail.argument_details):
            # Get parameter name by position from params dict
            param_name = param_names[i] if i < len(param_names) else f"arg_{i}"
            val = self._params.get(param_name, ad.default_value)
            if ad.type is PathKind.file:
                if isinstance(val, list):
                    paths = [str(path) for path in val if str(path).strip()]
                    if not paths:
                        continue
                    val = ", ".join(paths)
                elif not isinstance(val, str) or not val.strip():
                    continue
            if isinstance(val, Selection):
                opts = val.options if hasattr(val, "options") else []
                val = opts[0] if opts else "Select..."
            elif isinstance(val, dict):
                val = ", ".join(f"{k}: {v}" for k, v in val.items())
            parts.append(f"{ad.label}: {val}")
        return "\n".join(parts)

    def _extract_features_summary(self) -> str:
        raw_names = self._params.get("feature_names", "")
        if isinstance(raw_names, list):
            names = [str(name).strip() for name in raw_names if str(name).strip()]
        else:
            names = [part.strip() for part in str(raw_names).split(",") if part.strip()]
        if not names:
            return ""
        parts = ["Features: " + ", ".join(names)]
        if "autoencoder_latent" in names and "latent_dim" in self._params:
            parts.append(
                "Autoencoder: "
                f"latent {self._params.get('latent_dim')}, "
                f"epochs {self._params.get('num_epochs')}, "
                f"batch {self._params.get('batch_size')}, "
                f"learning rate {self._params.get('learning_rate')}"
            )
        return "\n".join(parts)

    def update_params(self, params: dict):
        self._params = dict(params)
        self._summary_label.setText(self._make_summary())

    def set_selected(self, selected: bool):
        self._apply_style(selected)

    def set_drag_enabled(self, enabled: bool) -> None:
        self._drag_enabled = enabled
        if not enabled:
            if self._dragging:
                self._end_drag(cancelled=True)
            else:
                self._press_pos = None

    def set_drag_contents_hidden(self, hidden: bool) -> None:
        if hidden:
            self._drag_saved_min_height = self.minimumHeight()
            self._drag_saved_max_height = self.maximumHeight()
            self.setFixedHeight(self.height())
            self._title.hide()
            self._edit_btn.hide()
            self._delete_btn.hide()
            self._summary_label.hide()
            self.setStyleSheet(
                "FunctionCardWidget { background: transparent; border: none;"
                " padding: 8px; margin: 2px 4px; }"
            )
            return
        self._title.show()
        self._edit_btn.show()
        self._delete_btn.show()
        self._summary_label.show()
        if self._drag_saved_min_height is not None:
            self.setMinimumHeight(self._drag_saved_min_height)
            self._drag_saved_min_height = None
        if self._drag_saved_max_height is not None:
            self.setMaximumHeight(self._drag_saved_max_height)
            self._drag_saved_max_height = None

    def _apply_style(self, selected: bool):
        if selected:
            self.setStyleSheet(
                "FunctionCardWidget { background: #cfe0fc;"
                " border: 2px solid #4285f4; border-radius: 6px;"
                " padding: 6px; margin: 2px 4px; color: #333333; }"
                " QLabel { color: #333333; background: transparent; }"
            )
        else:
            self.setStyleSheet(
                "FunctionCardWidget { background: white;"
                " border: 1px solid #ccc; border-radius: 6px;"
                " padding: 8px; margin: 2px 4px; color: #333333; }"
                " QLabel { color: #333333; background: transparent; }"
            )

    def enterEvent(self, event):
        if self._description and not self._dragging:
            self._tooltip_timer.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._tooltip_timer.stop()
        QToolTip.hideText()
        super().leaveEvent(event)

    def _show_delayed_tooltip(self):
        if self._description and self.underMouse() and not self._dragging:
            from PyQt5.QtGui import QCursor
            QToolTip.showText(QCursor.pos(), self._description, self)

    def _widget_is_action_button(self, widget: Optional[QWidget]) -> bool:
        while widget is not None and widget is not self:
            if widget is self._edit_btn or widget is self._delete_btn:
                return True
            widget = widget.parentWidget()
        return False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if (
                not self._drag_enabled
                or self._widget_is_action_button(self.childAt(event.pos()))
            ):
                self._press_pos = None
                self._dragging = False
            else:
                self._press_pos = QPoint(event.pos())
                self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self.drag_moved.emit(event.globalPos())
        elif (
            self._press_pos is not None
            and event.buttons() & Qt.LeftButton
        ):
            if (event.pos() - self._press_pos).manhattanLength() >= QApplication.startDragDistance():
                self._begin_drag()
                self.drag_moved.emit(event.globalPos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._dragging:
                self._end_drag(cancelled=False)
            else:
                self._press_pos = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if self._dragging and event.key() == Qt.Key_Escape:
            self._end_drag(cancelled=True)
            event.accept()
            return
        super().keyPressEvent(event)

    def _begin_drag(self) -> None:
        if not self._drag_enabled or self._dragging or self._press_pos is None:
            return
        self._dragging = True
        self._tooltip_timer.stop()
        QToolTip.hideText()
        self.grabMouse()
        self.grabKeyboard()
        self.drag_started.emit(self._index, QPoint(self._press_pos))

    def _end_drag(self, cancelled: bool) -> None:
        if not self._dragging:
            return
        self._dragging = False
        self._press_pos = None
        self.releaseMouse()
        self.releaseKeyboard()
        if cancelled:
            self.drag_cancelled.emit(self._index)
        else:
            self.drag_finished.emit(self._index)


class ParameterEditorWidget(QWidget):
    apply_clicked = pyqtSignal(dict)

    def __init__(self, manager: Manager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._manager = manager
        self._editors: list[tuple[str, QWidget, ArgumentDetail]] = []

        self.setStyleSheet("""
            QWidget { color: #333333; }
            QLabel { color: #333333; background: transparent; }
            QLineEdit { color: #333333; background: white; border: 1px solid #ccc; border-radius: 4px; padding: 4px 6px; }
            QComboBox { color: #333333; background: white; border: 1px solid #ccc; border-radius: 4px; padding: 4px 6px; }
            QComboBox QAbstractItemView { color: #333333; background: white; selection-background-color: #e8f0fe; selection-color: #111111; }
            QPushButton { color: #333333; }
            QPlainTextEdit { color: #333333; background: white; border: 1px solid #ccc; border-radius: 4px; padding: 4px 6px; }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        self._title = QLabel("<b>Edit Function Parameters</b>")
        self._title.setStyleSheet(
            "background: transparent; border: none;"
            " border-bottom: 1px solid #ccc; padding-bottom: 6px;"
        )
        layout.addWidget(self._title)

        self._form_container = QWidget()
        self._form_container.setStyleSheet("background: transparent;")
        self._form_layout = QFormLayout()
        self._form_layout.setLabelAlignment(Qt.AlignRight)
        self._form_container.setLayout(self._form_layout)
        layout.addWidget(self._form_container)

        self._apply_btn = QPushButton("Done")
        self._apply_btn.setStyleSheet(
            "QPushButton { background: white; color: #333; border: 1px solid #ccc;"
            " border-radius: 6px; padding: 8px 18px;"
            " font-size: 13px; font-weight: bold; }"
            " QPushButton:hover { background: #f5f5f5; border-color: #aaa; }"
        )
        self._apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(self._apply_btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.hide()

    def display(self, detail: FunctionDetail, current_params: dict, function_name: str = ""):
        self._clear()
        self._current_function_name = function_name
        self._current_detail = detail
        self._current_params_val = current_params # Keep reference to initial params
        
        # Special handling for model steps: model dropdown and training-option fields.
        if function_name in ["evaluate_model", "train_model", "evaluate_generic_model"]:
            model_combo = QComboBox()
            model_combo.setStyleSheet(self._COMBO_STYLE)
            model_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            model_combo.setMinimumWidth(120)
            
            model_names = []
            try:
                models = find_models()
                model_names = sorted(models.keys())
                for m_name in model_names:
                    model_combo.addItem(m_name)
            except Exception:
                warning = QLabel("Model list could not be loaded. Check dependencies.")
                warning.setWordWrap(True)
                warning.setStyleSheet("color: #b00020; font-size: 11px;")
                self._form_layout.addRow("", warning)
            
            current_model = current_params.get("model_name", "LDA")
            if isinstance(current_model, str) and current_model:
                idx = model_combo.findText(current_model)
                if idx >= 0:
                    model_combo.setCurrentIndex(idx)
                elif model_names:
                    model_combo.setCurrentIndex(0)

            import_errors = get_model_import_errors()
            if import_errors:
                skipped = ", ".join(sorted(import_errors.keys()))
                warning = QLabel(f"Some models are hidden because imports failed: {skipped}")
                warning.setWordWrap(True)
                warning.setStyleSheet("color: #8a5a00; font-size: 11px;")
                self._form_layout.addRow("", warning)
            
            self._editors.append(("model_name", model_combo, None))
            self._form_layout.addRow("Select Model:", model_combo)
            if function_name == "evaluate_generic_model":
                self._add_holdout_subjects_row(current_params)
            else:
                self._add_directory_path_rows(function_name, detail, current_params)

            # Connect change signal
            model_combo.currentTextChanged.connect(self._on_model_changed)
            
            # Trigger initial build of training options
            initial_opts = current_params.get("training_options", {})
            if function_name == "train_model":
                initial_opts = current_params.get("hyperparameters", {})
            self._refresh_training_options(model_combo.currentText(), initial_opts)
        elif function_name == "infer_on_model":
            options = current_params.get("training_options") or {}
            current_batch = options.get("batch_size", 32) if isinstance(options, dict) else 32
            batch_detail = ArgumentDetail(
                "batch_size", int, 32, "Number of trials per batch"
            )
            batch_edit = self._build_editor(batch_detail, current_batch)
            self._editors.append(("to_batch_size", batch_edit, batch_detail))
            self._form_layout.addRow("Batch Size:", batch_edit)
        elif function_name == "load_model":
            self._add_file_path_rows(function_name, detail, current_params)
        elif function_name == "extract_features":
            from ..features import FEATURE_REGISTRY

            available_features = list(FEATURE_REGISTRY.keys())
            current_val = current_params.get("feature_names", "")
            if isinstance(current_val, str):
                selected_features = {
                    name.strip()
                    for name in current_val.split(",")
                    if name.strip()
                }
            else:
                selected_features = set(current_val or [])

            feature_list = QListWidget()
            feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
            feature_list.setStyleSheet(
                "QListWidget { border: 1px solid #ccc; border-radius: 4px;"
                " background: white; color: #333333; }"
                " QListWidget::item { padding: 4px 8px; color: #333333; }"
                " QListWidget::item:selected { background: #4285f4; color: white; }"
                " QListWidget::item:hover { background: #f0f4ff; color: #333333; }"
            )
            for feature_name in available_features:
                feature_list.addItem(feature_name)
                if feature_name in selected_features:
                    feature_list.item(feature_list.count() - 1).setSelected(True)
            feature_list.setFixedHeight(min(36 * max(len(available_features), 1), 180))

            self._editors.append(("feature_names", feature_list, None))
            self._form_layout.addRow("Features:", feature_list)
            self._add_autoencoder_fields(current_params)
            feature_list.itemSelectionChanged.connect(self._sync_autoencoder_fields)
            self._sync_autoencoder_fields()
        elif function_name == "map_labels":
            self._add_file_path_rows(
                function_name, detail, current_params, self._browse_csv
            )
        else:
            # Standard positional matching for other functions
            param_names = self._parameter_names(function_name)
            for i, ad in enumerate(detail.argument_details):
                param_name = param_names[i] if i < len(param_names) else f"arg_{i}"
                current_val = current_params.get(param_name, ad.default_value)
                widget = self._build_editor(ad, current_val)
                self._editors.append((param_name, widget, ad))
                self._form_layout.addRow(f"{ad.label}:", widget)
                
        self.show()

    def _add_autoencoder_fields(self, current_params: dict) -> None:
        defaults = {
            "latent_dim": 128,
            "num_epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
        }
        fields = [
            ("latent_dim", "Latent Size", int),
            ("num_epochs", "Epochs", int),
            ("batch_size", "Batch Size", int),
            ("learning_rate", "Learning Rate", float),
        ]
        box = QWidget()
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        box.setLayout(form)
        for key, label, kind in fields:
            detail = ArgumentDetail(label, kind, defaults[key])
            editor = self._build_editor(detail, current_params.get(key, defaults[key]))
            self._editors.append((key, editor, detail))
            form.addRow(f"{label}:", editor)
        self._autoencoder_box = box
        self._form_layout.addRow(box)

    def _sync_autoencoder_fields(self) -> None:
        box = getattr(self, "_autoencoder_box", None)
        if box is None:
            return
        selected = False
        for name, widget, _ad in self._editors:
            if name == "feature_names" and isinstance(widget, QListWidget):
                selected = any(
                    item.text() == "autoencoder_latent" and item.isSelected()
                    for item in (widget.item(i) for i in range(widget.count()))
                )
                break
        box.setVisible(selected)

    def _add_holdout_subjects_row(self, current_params: dict) -> None:
        held_out = current_params.get("only_held_out")
        if isinstance(held_out, list):
            text = ", ".join(str(name) for name in held_out)
        elif isinstance(held_out, str):
            text = held_out
        else:
            text = ""

        holdout_edit = QLineEdit()
        holdout_edit.setStyleSheet(self._EDITOR_STYLE)
        holdout_edit.setText(text)
        holdout_edit.setPlaceholderText("Blank holds out every loaded subject")
        self._editors.append(("only_held_out", holdout_edit, None))
        self._form_layout.addRow("Hold out subjects:", holdout_edit)

    def _add_directory_path_rows(
        self,
        function_name: str,
        detail: FunctionDetail,
        current_params: dict,
    ) -> None:
        """Draw each directory-path argument from the function detail."""
        directory_details = [
            ad for ad in detail.argument_details if ad.type is PathKind.directory
        ]
        if not directory_details:
            return
        reserved = {"model_name", "training_options", "hyperparameters"}
        param_names = [
            name
            for name in self._parameter_names(function_name)
            if name not in reserved
        ]
        for ad, param_name in zip(directory_details, param_names):
            current = current_params.get(param_name, ad.default_value)
            if current is None:
                current = ""

            dir_edit = QLineEdit()
            dir_edit.setStyleSheet(self._EDITOR_STYLE)
            dir_edit.setText(str(current))

            choose_btn = QPushButton("Choose")
            choose_btn.setStyleSheet(
                "QPushButton { background: #e0e0e0; color: #333; border: 1px solid #ccc;"
                " border-radius: 4px; padding: 4px 8px; font-size: 11px; }"
                " QPushButton:hover { background: #d0d0d0; }"
            )
            choose_btn.clicked.connect(lambda _, le=dir_edit: self._browse_dir(le))

            dir_layout = QHBoxLayout()
            dir_layout.setContentsMargins(0, 0, 0, 0)
            dir_layout.addWidget(dir_edit)
            dir_layout.addWidget(choose_btn)
            dir_container = QWidget()
            dir_container.setLayout(dir_layout)

            self._editors.append((param_name, dir_edit, ad))
            self._form_layout.addRow(f"{ad.label}:", dir_container)

    def _add_file_path_rows(
        self,
        function_name: str,
        detail: FunctionDetail,
        current_params: dict,
        browse=None,
    ) -> None:
        """Draw each file-path argument from the function detail."""
        file_details = [
            ad for ad in detail.argument_details if ad.type is PathKind.file
        ]
        if not file_details:
            return
        if browse is None:
            browse = self._browse_file
        param_names = self._parameter_names(function_name)
        for ad, param_name in zip(file_details, param_names):
            current = current_params.get(param_name, ad.default_value)
            if current is None:
                current = ""
            elif isinstance(current, list):
                current = json.dumps(current)

            file_edit = QLineEdit()
            file_edit.setStyleSheet(self._EDITOR_STYLE)
            file_edit.setText(str(current))

            choose_btn = QPushButton("Choose")
            choose_btn.setStyleSheet(
                "QPushButton { background: #e0e0e0; color: #333; border: 1px solid #ccc;"
                " border-radius: 4px; padding: 4px 8px; font-size: 11px; }"
                " QPushButton:hover { background: #d0d0d0; }"
            )
            choose_btn.clicked.connect(lambda _, le=file_edit, browse=browse: browse(le))

            file_layout = QHBoxLayout()
            file_layout.setContentsMargins(0, 0, 0, 0)
            file_layout.addWidget(file_edit)
            file_layout.addWidget(choose_btn)
            file_container = QWidget()
            file_container.setLayout(file_layout)

            self._editors.append((param_name, file_edit, ad))
            self._form_layout.addRow(f"{ad.label}:", file_container)

    def _parameter_names(self, function_name: str) -> list[str]:
        try:
            func = self._manager.find_functions().get(function_name)
            if func is None:
                return []
            sig = inspect.signature(func)
            return [name for name in sig.parameters.keys() if name != "self"]
        except Exception:
            return []

    def _browse_csv(self, line_edit):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            line_edit.setText(path)

    def _browse_file(self, line_edit):
        from PyQt5.QtWidgets import QFileDialog
        import json
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Model File(s)", "", "Pickle Files (*.pkl);;All Files (*)")
        if paths:
            if len(paths) == 1:
                line_edit.setText(paths[0])
            else:
                line_edit.setText(json.dumps(paths))
                
    def _browse_dir(self, line_edit):
        from PyQt5.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            line_edit.setText(path)

    def _on_model_changed(self, model_name: str):
        # We need to rebuild the training_options part of the form
        # First, find and remove existing training_options editors
        to_remove = []
        for i, (name, widget, ad) in enumerate(self._editors):
            if name == "training_options" or name.startswith("to_"):
                to_remove.append(i)
        
        for i in reversed(to_remove):
            name, widget, ad = self._editors.pop(i)
            # Find the row in the form layout
            for row in range(self._form_layout.rowCount()):
                f_item = self._form_layout.itemAt(row, QFormLayout.FieldRole)
                if f_item and f_item.widget() == widget:
                    self._form_layout.removeRow(row)
                    break
        
        # Now add new ones
        self._refresh_training_options(model_name, {})

    def _refresh_training_options(self, model_name: str, current_val: dict):
        # Hardcoded model parameters since they are not in the model classes anymore
        model_name_lower = model_name.lower()
        
        if model_name_lower == "jason_cnn":
            ads = [
                ArgumentDetail("num_epochs", int, 15, "Number of training epochs"),
                ArgumentDetail("batch_size", int, 32, "Number of trials per batch"),
                ArgumentDetail("learning_rate", float, 1e-3, "Optimizer learning rate"),
                ArgumentDetail("val_split", float, 0.10, "Validation split ratio"),
                ArgumentDetail("l2", float, 1e-5, "L2 regularization penalty"),
                ArgumentDetail("sdrop", float, 0.10, "Spatial dropout rate"),
                ArgumentDetail("head_drop", float, 0.30, "Head dropout rate"),
                ArgumentDetail("n_classes", int, 4, "Number of output classes"),
            ]
        elif model_name_lower == "cnn":
            ads = [
                ArgumentDetail("num_epochs", int, 20, "Number of training epochs"),
                ArgumentDetail("batch_size", int, 32, "Number of trials per batch"),
                ArgumentDetail("learning_rate", float, 1e-3, "Optimizer learning rate"),
                ArgumentDetail("weight_decay", float, 0.1, "L2 regularization penalty"),
                ArgumentDetail("n_classes", int, 4, "Number of output classes"),
                ArgumentDetail("p_drop", float, 0.1, "Dropout probability"),
            ]
        elif model_name_lower in ["pitchcnn", "autocorrcnn", "zerocrossingcnn"]:
            ads = [
                ArgumentDetail("num_epochs", int, 20, "Number of training epochs"),
                ArgumentDetail("batch_size", int, 32, "Number of trials per batch"),
                ArgumentDetail("learning_rate", float, 1e-3, "Optimizer learning rate"),
                ArgumentDetail("weight_decay", float, 0.1, "L2 regularization penalty"),
                ArgumentDetail("n_classes", int, 4, "Number of output classes"),
                ArgumentDetail("p_drop", float, 0.1, "Dropout probability"),
            ]
        elif model_name_lower == "multibranchcnn":
            ads = [
                ArgumentDetail("feature_inputs", str, "raw,pitchtrack", "Comma-separated branches: raw, pitchtrack, autocorr, zerocrossing"),
                ArgumentDetail("num_epochs", int, 20, "Number of training epochs"),
                ArgumentDetail("batch_size", int, 32, "Number of trials per batch"),
                ArgumentDetail("learning_rate", float, 1e-3, "Optimizer learning rate"),
                ArgumentDetail("weight_decay", float, 0.1, "L2 regularization penalty"),
                ArgumentDetail("n_classes", int, 4, "Number of output classes"),
                ArgumentDetail("p_drop", float, 0.1, "Dropout probability"),
            ]
        elif model_name_lower == "svm":
            ads = [
                ArgumentDetail("search_type", str, "grid", "Search type: 'grid' or 'random'"),
                ArgumentDetail("n_iter", int, 20, "Number of iterations for random search"),
            ]
        elif model_name_lower == "lda":
            ads = [
                ArgumentDetail("solver", str, "lsqr", "Solver: 'svd', 'lsqr', or 'eigen'"),
                ArgumentDetail("shrinkage", str, "auto", "Shrinkage: 'auto' or float value"),
            ]
        elif model_name_lower in ["lstm", "rnn", "gru", "ffnn", "transformer"]:
            ads = [
                ArgumentDetail("num_epochs", int, 20, "Number of training epochs"),
                ArgumentDetail("batch_size", int, 32, "Number of trials per batch"),
                ArgumentDetail("learning_rate", float, 1e-3, "Optimizer learning rate"),
                ArgumentDetail("weight_decay", float, 0.1, "L2 regularization penalty"),
            ]
        elif model_name_lower in [
            "dynamiccnn",
            "dynamicffnn",
            "dynamicrnn",
            "dynamiccrnn",
            "dynamictransformer",
            "multibranchffnn",
            "multichannelcnn",
            "transformermulti",
        ]:
            ads = [
                ArgumentDetail("num_epochs", int, 50, "Number of training epochs"),
                ArgumentDetail("batch_size", int, 32, "Number of trials per batch"),
                ArgumentDetail("learning_rate", float, 1e-3, "Optimizer learning rate"),
                ArgumentDetail("weight_decay", float, 0.1, "L2 regularization penalty"),
                ArgumentDetail("patience", int, 20, "Early stopping patience"),
                ArgumentDetail("min_delta", float, 0.001, "Minimum improvement for early stopping"),
                ArgumentDetail("embed_dim", int, 64, "Embedding dimension per branch"),
            ]
        elif model_name_lower in ["autoencoder", "autoencoder_loso"]:
            ads = [
                ArgumentDetail("num_epochs", int, 100, "Number of training epochs"),
                ArgumentDetail("batch_size", int, 64, "Number of trials per batch"),
                ArgumentDetail("learning_rate", float, 1e-3, "Optimizer learning rate"),
                ArgumentDetail("latent_dim", int, 128, "Latent space dimensions"),
            ]
        else:
            # Fallback to default generic dict
            ads = [ArgumentDetail("training_options", dict, {}, "Training options")]
        
        # If it's a list of ads, we want to create a sub-form or a specialized dict editor
        # The user wants "what parameters can be edit" to change.
        # Let's create individual fields for each AD in the list, but they will all
        # be collected into the 'training_options' dict.
        
        # We'll use a special container or just add them to the main form with a prefix
        for ad in ads:
            val = current_val.get(ad.label if ad.label else "", ad.default_value)
            # If the label is used as key
            key = ad.label # Usually the label is the key in these specialized ads
            
            widget = self._build_editor(ad, current_val.get(key, ad.default_value))
            
            if model_name_lower == "lda" and key in ["solver", "shrinkage"]:
                widget.setEnabled(False)
                
            # We'll tag these as 'training_options_part' so we can collect them later
            self._editors.append((f"to_{key}", widget, ad))
            self._form_layout.addRow(f"{key}:", widget)

    _EDITOR_STYLE = (
        "border: 1px solid #ccc; border-radius: 4px;"
        " padding: 4px 6px; background: white; color: #333333;"
    )

    _TEXTEDIT_STYLE = (
        "QPlainTextEdit { border: 1px solid #ccc; border-radius: 4px;"
        " padding: 4px 6px; background: white; color: #333333; }"
        " QPlainTextEdit QScrollBar:vertical {"
        "   background: #f0f0f0; width: 8px; border-radius: 4px;"
        " }"
        " QPlainTextEdit QScrollBar::handle:vertical {"
        "   background: #c0c0c0; border-radius: 4px; min-height: 20px;"
        " }"
        " QPlainTextEdit QScrollBar::handle:vertical:hover {"
        "   background: #a0a0a0;"
        " }"
        " QPlainTextEdit QScrollBar::add-line:vertical,"
        " QPlainTextEdit QScrollBar::sub-line:vertical {"
        "   height: 0px;"
        " }"
    )

    _COMBO_STYLE = (
        "QComboBox { border: 1px solid #ccc; border-radius: 4px;"
        " padding: 4px 6px; background: white; color: #333333; }"
        " QComboBox QAbstractItemView {"
        "   background: white; color: #333;"
        "   selection-background-color: #e8f0fe;"
        "   selection-color: #111;"
        "   outline: none;"
        " }"
        " QComboBox QAbstractItemView::item:hover {"
        "   background: #e8f0fe; color: #111;"
        " }"
        " QComboBox QAbstractItemView::item:selected {"
        "   background: #d2e3fc; color: #111;"
        " }"
    )

    def _build_editor(self, ad: ArgumentDetail, current_value: Any) -> QWidget:
        typ = ad.type
        if typ is int:
            w = QLineEdit()
            w.setStyleSheet(self._EDITOR_STYLE)
            w.setValidator(QIntValidator(-10_000_000, 10_000_000, self))
            if current_value is not None:
                w.setText(str(current_value))
            if ad.description:
                w.setPlaceholderText(ad.description)
            return w
        if typ is float:
            w = QLineEdit()
            w.setStyleSheet(self._EDITOR_STYLE)
            validator = QDoubleValidator(-1e9, 1e9, 6, self)
            validator.setNotation(QDoubleValidator.StandardNotation)
            w.setValidator(validator)
            if current_value is not None:
                w.setText(str(current_value))
            if ad.description:
                w.setPlaceholderText(ad.description)
            return w
        if typ is str:
            w = QLineEdit()
            w.setStyleSheet(self._EDITOR_STYLE)
            if current_value is not None:
                w.setText(str(current_value))
            if ad.description:
                w.setPlaceholderText(ad.description)
            return w
        if _is_dict_type(typ):
            w = DictEditorWidget(current_value if isinstance(current_value, dict) else {})
            return w
        if typ is Selection:
            combo = QComboBox()
            combo.setStyleSheet(self._COMBO_STYLE)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            combo.setMinimumWidth(120)
            selection = (
                current_value
                if isinstance(current_value, Selection)
                else ad.default_value
            )
            if isinstance(selection, Selection):
                try:
                    try:
                        mapping = selection.option_value_map(self._manager.state)
                    except TypeError:
                        mapping = selection.option_value_map()
                    for opt in mapping.keys():
                        combo.addItem(opt)
                    if isinstance(current_value, str):
                        idx = combo.findText(current_value)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                except Exception as e:
                    # If loading options fails (e.g., PyTorch error), provide text input instead
                    text_input = QLineEdit()
                    text_input.setStyleSheet(self._EDITOR_STYLE)
                    text_input.setPlaceholderText("(Unable to load options - enter model name manually)")
                    if isinstance(current_value, str):
                        text_input.setText(current_value)
                    return text_input
            return combo
        w = QLineEdit()
        w.setStyleSheet(self._EDITOR_STYLE)
        if current_value is not None:
            w.setText(str(current_value))
        return w

    def _on_apply(self):
        try:
            params = self._collect()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Parameters", str(exc))
            return
        self.apply_clicked.emit(params)

    def _collect(self) -> dict:
        result: dict[str, Any] = {}
        training_options = {}
        
        for name, widget, _ad in self._editors:
            if not widget.isVisible():
                continue
            val = None
            if isinstance(widget, QLineEdit):
                val = widget.text()
                if _ad is not None and _ad.type is PathKind.directory:
                    val = val.strip() or None
                elif _ad is not None and _ad.type is PathKind.file:
                    if self._current_function_name == "map_labels":
                        val = val.strip()
                    else:
                        val = self._file_path_value(val)
                elif _ad is not None and _ad.type is int:
                    if val.strip() == "":
                        raise ValueError(f"{_ad.label} requires an integer value.")
                    val = int(val)
                elif _ad is not None and _ad.type is float:
                    if val.strip() == "":
                        raise ValueError(f"{_ad.label} requires a numeric value.")
                    val = float(val)
                elif val.strip().startswith("[") and val.strip().endswith("]"):
                    try:
                        val = json.loads(val.strip())
                    except Exception:
                        pass
            elif isinstance(widget, QPlainTextEdit):
                text = widget.toPlainText().strip()
                if not text:
                    val = {}
                else:
                    try:
                        val = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON: {exc}") from exc
            elif isinstance(widget, DictEditorWidget):
                val = widget.get_value()
            elif isinstance(widget, QComboBox):
                val = widget.currentText()
            elif isinstance(widget, QListWidget):
                val = ",".join(item.text() for item in widget.selectedItems())
            
            if name.startswith("to_"):
                # Re-assemble training_options
                key = name[3:]
                training_options[key] = val
            else:
                result[name] = val
        
        if self._current_function_name == "infer_on_model":
            batch_size = training_options.get("batch_size")
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("Batch Size requires a positive integer.")
            return {"training_options": {"batch_size": batch_size}}
        if self._current_function_name == "evaluate_model" and training_options:
            result["training_options"] = training_options
        elif self._current_function_name == "train_model" and training_options:
            result["hyperparameters"] = training_options
        elif self._current_function_name == "extract_features":
            names = [
                part.strip()
                for part in str(result.get("feature_names", "")).split(",")
                if part.strip()
            ]
            if not names:
                raise ValueError("Select at least one feature.")
            stored = {"feature_names": ",".join(names)}
            if "autoencoder_latent" in names:
                latent_dim = result.get("latent_dim")
                num_epochs = result.get("num_epochs")
                batch_size = result.get("batch_size")
                learning_rate = result.get("learning_rate")
                if not isinstance(latent_dim, int) or latent_dim <= 0:
                    raise ValueError("Latent Size requires a positive integer.")
                if not isinstance(num_epochs, int) or num_epochs <= 0:
                    raise ValueError("Epochs requires a positive integer.")
                if not isinstance(batch_size, int) or batch_size <= 0:
                    raise ValueError("Batch Size requires a positive integer.")
                if not isinstance(learning_rate, float) or learning_rate <= 0:
                    raise ValueError("Learning Rate requires a positive number.")
                stored.update({
                    "latent_dim": latent_dim,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                })
            return stored
        elif self._current_function_name == "evaluate_generic_model":
            if training_options:
                result["training_options"] = training_options
            result["only_held_out"] = self._holdout_names(result.get("only_held_out"))
            
        return result

    @staticmethod
    def _holdout_names(value) -> list[str] | None:
        """A blank hold-out field is no list. Names are stored as a list."""
        if value is None:
            return None
        if isinstance(value, list):
            names = [str(name).strip() for name in value if str(name).strip()]
        else:
            names = [part.strip() for part in str(value).split(",") if part.strip()]
        return names or None

    @staticmethod
    def _file_path_value(text: str) -> str | list:
        """One chosen file is a string. Several are a list. A blank field stays blank."""
        stripped = text.strip()
        if not stripped:
            return ""
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
            if isinstance(parsed, list):
                return parsed
        return stripped

    def _clear(self):
        for i in reversed(range(self._form_layout.count())):
            item = self._form_layout.takeAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        self._editors.clear()

    def clear_and_hide(self):
        self._clear()
        self.hide()


class PngSizeDialog(QDialog):
    """Ask for a PNG width and height in pixels before a save dialog opens."""

    def __init__(self, parent=None, width: int = 1920, height: int = 1080, title: str = "PNG Size"):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle(title)
        self.setModal(True)

        self._width_edit = QLineEdit(str(width))
        self._height_edit = QLineEdit(str(height))
        validator = QIntValidator(1, 100_000, self)
        self._width_edit.setValidator(validator)
        self._height_edit.setValidator(validator)
        self._width_edit.textChanged.connect(self._update_okay)
        self._height_edit.textChanged.connect(self._update_okay)

        layout = QVBoxLayout()
        layout.addLayout(self._labeled_row("Height (px)", self._height_edit))
        layout.addLayout(self._labeled_row("Width (px)", self._width_edit))

        buttons = QHBoxLayout()
        buttons.addStretch()
        self._okay_btn = QPushButton("Okay")
        self._okay_btn.setDefault(True)
        self._okay_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self._okay_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)
        self.setLayout(layout)
        self._update_okay()

    @staticmethod
    def _labeled_row(label: str, field: QLineEdit) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(field, stretch=1)
        return row

    def _update_okay(self) -> None:
        self._okay_btn.setEnabled(
            self._positive_int(self._width_edit.text()) is not None
            and self._positive_int(self._height_edit.text()) is not None
        )

    @staticmethod
    def _positive_int(text: str) -> int | None:
        stripped = text.strip()
        if not stripped.isdigit():
            return None
        value = int(stripped)
        if value <= 0:
            return None
        return value

    def size_px(self) -> tuple[int, int]:
        width = self._positive_int(self._width_edit.text())
        height = self._positive_int(self._height_edit.text())
        if width is None or height is None:
            raise ValueError("Width and height must be positive integers.")
        return width, height
