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

    def display(self, detail: FunctionDetail, current_params: dict):
        self._clear()
        # Get parameter names from current_params dict
        param_names = list(current_params.keys())
        
        for i, ad in enumerate(detail.argument_details):
            # Get parameter name by position
            param_name = param_names[i] if i < len(param_names) else f"arg_{i}"
            current_val = current_params.get(param_name, ad.default_value)
            widget = self._build_editor(ad, current_val)
            self._editors.append((param_name, widget, ad))
            self._form_layout.addRow(f"{ad.label}:", widget)
        self.show()

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
        for name, widget, _ad in self._editors:
            if isinstance(widget, QSpinBox):
                result[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                result[name] = widget.value()
            elif isinstance(widget, QLineEdit):
                result[name] = widget.text()
            elif isinstance(widget, QPlainTextEdit):
                text = widget.toPlainText().strip()
                if not text:
                    result[name] = {}
                else:
                    try:
                        result[name] = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON: {exc}") from exc
            elif isinstance(widget, DictEditorWidget):
                result[name] = widget.get_value()
            elif isinstance(widget, QComboBox):
                result[name] = widget.currentText()
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