from __future__ import annotations

import json
from typing import Any, Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .manager import Manager
from ..core.utils.function_detail import ArgumentDetail, FunctionDetail, Selection
from ..models.utils import find_model, find_models


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

        self.setFrameShape(QFrame.Box)
        self.setLineWidth(1)
        self._apply_style(selected=False)
        self.setMouseTracking(True)

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
        header.addWidget(self._title)
        header.addStretch()

        edit_btn = QPushButton("Edit")
        edit_btn.setFixedSize(50, 24)
        edit_btn.setStyleSheet(
            "QPushButton { background: #e0e0e0; color: #333; border: 1px solid #ccc;"
            " border-radius: 4px; font-size: 11px; }"
            " QPushButton:hover { background: #d0d0d0; }"
        )
        edit_btn.clicked.connect(lambda: self.edit_clicked.emit(self._index))
        header.addWidget(edit_btn)

        delete_btn = QPushButton("Delete")
        delete_btn.setFixedSize(50, 24)
        delete_btn.setStyleSheet(
            "QPushButton { background: #e0e0e0; color: #c62828; border: 1px solid #ccc;"
            " border-radius: 4px; font-size: 11px; }"
            " QPushButton:hover { background: #ffcdd2; }"
        )
        delete_btn.clicked.connect(lambda: self.delete_clicked.emit(self._index))
        header.addWidget(delete_btn)
        layout.addLayout(header)

        self._summary_label = QLabel(self._make_summary())
        self._summary_label.setStyleSheet("color: #555; font-size: 11px;")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self.setLayout(layout)

    def _make_summary(self) -> str:
        if not self._detail or not self._detail.argument_details:
            return ""
        parts = []
        param_names = list(self._params.keys())
        
        for i, ad in enumerate(self._detail.argument_details):
            # Get parameter name by position from params dict
            param_name = param_names[i] if i < len(param_names) else f"arg_{i}"
            val = self._params.get(param_name, ad.default_value)
            if isinstance(val, Selection):
                opts = val.options if hasattr(val, "options") else []
                val = opts[0] if opts else "Select..."
            elif isinstance(val, dict):
                val = ", ".join(f"{k}: {v}" for k, v in val.items())
            parts.append(f"{ad.label}: {val}")
        return "\n".join(parts)

    def update_params(self, params: dict):
        self._params = dict(params)
        self._summary_label.setText(self._make_summary())

    def set_selected(self, selected: bool):
        self._apply_style(selected)

    def _apply_style(self, selected: bool):
        if selected:
            self.setStyleSheet(
                "FunctionCardWidget { background: #cfe0fc;"
                " border: 2px solid #4285f4; border-radius: 6px;"
                " padding: 6px; margin: 2px 4px; }"
            )
        else:
            self.setStyleSheet(
                "FunctionCardWidget { background: white;"
                " border: 1px solid #ccc; border-radius: 6px;"
                " padding: 8px; margin: 2px 4px; }"
            )

    def enterEvent(self, event):
        if self._description:
            self._tooltip_timer.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._tooltip_timer.stop()
        QToolTip.hideText()
        super().leaveEvent(event)

    def _show_delayed_tooltip(self):
        if self._description and self.underMouse():
            from PyQt5.QtGui import QCursor
            QToolTip.showText(QCursor.pos(), self._description, self)


class ParameterEditorWidget(QWidget):
    apply_clicked = pyqtSignal(dict)

    def __init__(self, manager: Manager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._manager = manager
        self._editors: list[tuple[str, QWidget, ArgumentDetail]] = []

        self.setStyleSheet("background: transparent;")

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

        self._apply_btn = QPushButton("Apply Changes")
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
        
        # Special handling for evaluate_model and train_model: add model_name dropdown
        if function_name in ["evaluate_model", "train_model"]:
            model_combo = QComboBox()
            model_combo.setStyleSheet(self._COMBO_STYLE)
            model_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            model_combo.setMinimumWidth(120)
            
            try:
                models = find_models()
                for m_name in sorted(models.keys()):
                    model_combo.addItem(m_name)
            except Exception:
                pass
            
            current_model = current_params.get("model_name", "LDA")
            if isinstance(current_model, str) and current_model:
                idx = model_combo.findText(current_model)
                if idx >= 0:
                    model_combo.setCurrentIndex(idx)
            
            self._editors.append(("model_name", model_combo, None))
            self._form_layout.addRow("Select Model:", model_combo)
            
            if function_name == "train_model":
                dir_container = QWidget()
                dir_layout = QHBoxLayout()
                dir_layout.setContentsMargins(0, 0, 0, 0)
                
                dir_edit = QLineEdit()
                dir_edit.setStyleSheet(self._EDITOR_STYLE)
                dir_edit.setText(current_params.get("output_dirpath", ""))
                
                browse_btn = QPushButton("Browse")
                browse_btn.setStyleSheet(
                    "QPushButton { background: #e0e0e0; color: #333; border: 1px solid #ccc;"
                    " border-radius: 4px; padding: 4px 8px; font-size: 11px; }"
                    " QPushButton:hover { background: #d0d0d0; }"
                )
                browse_btn.clicked.connect(lambda _, le=dir_edit: self._browse_dir(le))
                
                dir_layout.addWidget(dir_edit)
                dir_layout.addWidget(browse_btn)
                dir_container.setLayout(dir_layout)
                
                self._editors.append(("output_dirpath", dir_edit, None))
                self._form_layout.addRow("Output Directory:", dir_container)
            
            # Connect change signal
            model_combo.currentTextChanged.connect(self._on_model_changed)
            
            # Trigger initial build of training options
            initial_opts = current_params.get("training_options", {})
            if function_name == "train_model":
                initial_opts = current_params.get("hyperparameters", {})
            self._refresh_training_options(model_combo.currentText(), initial_opts)
        elif function_name == "load_model":
            param_name = "model_pickle_filepath"
            current_val = current_params.get(param_name, "")
            
            container = QWidget()
            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            
            line_edit = QLineEdit()
            line_edit.setStyleSheet(self._EDITOR_STYLE)
            if current_val:
                line_edit.setText(str(current_val))
                
            browse_btn = QPushButton("Browse")
            browse_btn.setStyleSheet(
                "QPushButton { background: #e0e0e0; color: #333; border: 1px solid #ccc;"
                " border-radius: 4px; padding: 4px 8px; font-size: 11px; }"
                " QPushButton:hover { background: #d0d0d0; }"
            )
            browse_btn.clicked.connect(lambda _, le=line_edit: self._browse_file(le))
            
            layout.addWidget(line_edit)
            layout.addWidget(browse_btn)
            container.setLayout(layout)
            
            self._editors.append((param_name, line_edit, detail.argument_details[0]))
            self._form_layout.addRow("Model Filepath:", container)
        else:
            # Standard positional matching for other functions
            for i, ad in enumerate(detail.argument_details):
                param_names = list(current_params.keys())
                param_name = param_names[i] if i < len(param_names) else f"arg_{i}"
                current_val = current_params.get(param_name, ad.default_value)
                widget = self._build_editor(ad, current_val)
                self._editors.append((param_name, widget, ad))
                self._form_layout.addRow(f"{ad.label}:", widget)
                
        self.show()

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
        " padding: 4px 6px; background: white;"
    )

    _TEXTEDIT_STYLE = (
        "QPlainTextEdit { border: 1px solid #ccc; border-radius: 4px;"
        " padding: 4px 6px; background: white; }"
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
        " padding: 4px 6px; background: white; }"
        " QComboBox QAbstractItemView {"
        "   background: white; color: #333;"
        "   selection-background-color: #4285f4;"
        "   selection-color: white;"
        "   outline: none;"
        " }"
    )

    def _build_editor(self, ad: ArgumentDetail, current_value: Any) -> QWidget:
        typ = ad.type
        if typ is int:
            w = QSpinBox()
            w.setStyleSheet(self._EDITOR_STYLE)
            w.setMinimum(-10_000_000)
            w.setMaximum(10_000_000)
            if current_value is not None:
                w.setValue(int(current_value))
            return w
        if typ is float:
            w = QDoubleSpinBox()
            w.setStyleSheet(self._EDITOR_STYLE)
            w.setDecimals(6)
            w.setMinimum(-1e9)
            w.setMaximum(1e9)
            if current_value is not None:
                w.setValue(float(current_value))
            return w
        if typ is str:
            w = QLineEdit()
            w.setStyleSheet(self._EDITOR_STYLE)
            if current_value is not None:
                w.setText(str(current_value))
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
            val = None
            if isinstance(widget, QSpinBox):
                val = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                val = widget.value()
            elif isinstance(widget, QLineEdit):
                val = widget.text()
                if val.strip().startswith("[") and val.strip().endswith("]"):
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
            
            if name.startswith("to_"):
                # Re-assemble training_options
                key = name[3:]
                training_options[key] = val
            else:
                result[name] = val
        
        if self._current_function_name == "evaluate_model" and training_options:
            result["training_options"] = training_options
        elif self._current_function_name == "train_model" and training_options:
            result["hyperparameters"] = training_options
            
        return result

    def _clear(self):
        for i in reversed(range(self._form_layout.count())):
            item = self._form_layout.takeAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        self._editors.clear()

    def clear_and_hide(self):
        self._clear()
        self.hide()