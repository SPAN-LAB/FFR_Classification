from __future__ import annotations

import json
from typing import Any, Optional

from PyQt5.QtCore import Qt, pyqtSignal
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
    QVBoxLayout,
    QWidget,
)

from .manager import Manager
from ..core.utils.function_detail import ArgumentDetail, FunctionDetail, Selection


def _is_dict_type(typ) -> bool:
    if typ is dict:
        return True
    return getattr(typ, "__origin__", None) is dict


class FunctionCardWidget(QFrame):
    edit_clicked = pyqtSignal(int)

    _BASE_STYLE = """
        FunctionCardWidget {{
            background: {bg};
            border: {border};
            border-radius: 6px;
            padding: 8px;
            margin: 2px 4px;
        }}
    """

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

        self.setFrameShape(QFrame.StyledPanel)
        self._apply_style(selected=False)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self._title = QLabel(f"<b>{label}</b>")
        self._title.setStyleSheet("border: none;")
        header.addWidget(self._title)
        header.addStretch()

        edit_btn = QPushButton("Edit")
        edit_btn.setFixedSize(50, 24)
        edit_btn.setStyleSheet(
            "QPushButton { background: #e8e8e8; border: 1px solid #bbb;"
            " border-radius: 4px; font-size: 11px; }"
            " QPushButton:hover { background: #d0d0d0; }"
        )
        edit_btn.clicked.connect(lambda: self.edit_clicked.emit(self._index))
        header.addWidget(edit_btn)
        layout.addLayout(header)

        self._summary_label = QLabel(self._make_summary())
        self._summary_label.setStyleSheet(
            "color: #555; font-size: 11px; border: none;"
        )
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self.setLayout(layout)

    def _make_summary(self) -> str:
        if not self._detail or not self._detail.argument_details:
            return ""
        parts = []
        for ad in self._detail.argument_details:
            val = self._params.get(ad.name, ad.default_value)
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
                self._BASE_STYLE.format(bg="#cfe0fc", border="2px solid #4285f4")
            )
        else:
            self.setStyleSheet(
                self._BASE_STYLE.format(bg="white", border="1px solid #ccc")
            )


class ParameterEditorWidget(QWidget):
    apply_clicked = pyqtSignal(dict)

    def __init__(self, manager: Manager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._manager = manager
        self._editors: list[tuple[str, QWidget, ArgumentDetail]] = []

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 8, 4, 4)

        self._title = QLabel("<b>Edit Function Parameters</b>")
        layout.addWidget(self._title)

        self._form_container = QWidget()
        self._form_layout = QFormLayout()
        self._form_layout.setLabelAlignment(Qt.AlignRight)
        self._form_container.setLayout(self._form_layout)
        layout.addWidget(self._form_container)

        self._apply_btn = QPushButton("Apply Changes")
        self._apply_btn.setStyleSheet(
            "QPushButton { background: #4285f4; color: white; border: none;"
            " border-radius: 4px; padding: 6px 18px; font-weight: bold; }"
            " QPushButton:hover { background: #3367d6; }"
        )
        self._apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(self._apply_btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.hide()

    def display(self, detail: FunctionDetail, current_params: dict):
        self._clear()
        for ad in detail.argument_details:
            current_val = current_params.get(ad.name, ad.default_value)
            widget = self._build_editor(ad, current_val)
            self._editors.append((ad.name, widget, ad))
            self._form_layout.addRow(f"{ad.label}:", widget)
        self.show()

    def _build_editor(self, ad: ArgumentDetail, current_value: Any) -> QWidget:
        typ = ad.type
        if typ is int:
            w = QSpinBox()
            w.setMinimum(-10_000_000)
            w.setMaximum(10_000_000)
            if current_value is not None:
                w.setValue(int(current_value))
            return w
        if typ is float:
            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setMinimum(-1e9)
            w.setMaximum(1e9)
            if current_value is not None:
                w.setValue(float(current_value))
            return w
        if typ is str:
            w = QLineEdit()
            if current_value is not None:
                w.setText(str(current_value))
            return w
        if _is_dict_type(typ):
            w = QPlainTextEdit()
            w.setMaximumHeight(100)
            w.setPlaceholderText("Enter JSON object")
            if isinstance(current_value, dict):
                w.setPlainText(json.dumps(current_value, indent=2))
            return w
        if typ is Selection:
            combo = QComboBox()
            selection = (
                current_value
                if isinstance(current_value, Selection)
                else ad.default_value
            )
            if isinstance(selection, Selection):
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
            return combo
        w = QLineEdit()
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
