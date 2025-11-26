from PyQt5.QtWidgets import (
    QWidget, QLabel, QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy, QApplication,
    QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox, QFormLayout
)
from PyQt5.QtCore import Qt, QMimeData, pyqtSignal
from PyQt5.QtGui import QPalette, QColor, QDrag, QPixmap, QPainter

from ...core.utils import FunctionDetail, Selection
from .flow_layout import FlowLayout

class FunctionBlock(QWidget):
    clicked = pyqtSignal(object)

    def __init__(self, parent, function_detail: FunctionDetail):
        super().__init__(parent)
        self.function_detail = function_detail
        self._selected = False
        
        # Allow widget to size based on content (minimum size possible)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        # Style for rounded rectangle
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.update_style()

        # Main layout (Horizontal) to hold header and args side by side
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.setSizeConstraint(QHBoxLayout.SizeConstraint.SetMinimumSize)

        # --- Left Side (Label) ---
        label_container = QWidget()
        label_container.setFixedWidth(120)
        label_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        
        label_layout = QVBoxLayout(label_container)
        label_layout.setContentsMargins(5, 5, 5, 5)
        label_layout.setSpacing(0)
        label_layout.setAlignment(Qt.AlignCenter)
        label_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)

        # Label
        self.label = QLabel(function_detail.label)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setMaximumWidth(110)  # Slightly less than container width for padding
        self.label.setScaledContents(False)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(36, 36, 36))
        self.label.setPalette(pal)
        
        label_layout.addWidget(self.label)
        main_layout.addWidget(label_container)

        # --- Divider ---
        v_line = QFrame()
        v_line.setFrameShape(QFrame.VLine)
        v_line.setFrameShadow(QFrame.Plain)
        v_line.setLineWidth(1)
        v_line.setStyleSheet("color: gray;")
        main_layout.addWidget(v_line)

        # --- Right Side (Arguments) ---
        args_container = QWidget()
        args_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Use FlowLayout for wrapping arguments
        args_layout = FlowLayout(args_container, margin=10, hSpacing=15, vSpacing=5)
        
        if function_detail.argument_details:
            for arg in function_detail.argument_details:
                self._add_argument_ui(arg, args_layout)
        # FlowLayout handles spacing, no need for stretch
        
        main_layout.addWidget(args_container)

    def _add_argument_ui(self, arg, layout):
        # Create a widget that holds "Label Input" pair
        # This widget will be an item in the FlowLayout
        pair_widget = QWidget()
        pair_layout = QHBoxLayout(pair_widget)
        pair_layout.setContentsMargins(0, 0, 0, 0)
        pair_layout.setSpacing(5)
        
        # Label
        label = QLabel(arg.label)
        label.setStyleSheet("color: #dddddd; font-weight: bold;")
        pair_layout.addWidget(label)
        
        # Input Widget
        widget = self._create_input_widget(arg.type, arg.default_value)
        if widget:
            pair_layout.addWidget(widget)
            
        layout.addWidget(pair_widget)

    def _create_input_widget(self, arg_type, default_value):
        # Handle Selection
        if arg_type == Selection or isinstance(default_value, Selection):
            combo = QComboBox()
            combo.setFixedWidth(100)
            options = []
            if isinstance(default_value, Selection):
                try:
                    options = default_value.options
                except Exception as e:
                    print(f"Error getting options for selection: {e}")
            combo.addItems(options)
            return combo

        # Handle Dict (Recursive / Nested)
        if (isinstance(arg_type, type) and issubclass(arg_type, dict)) or isinstance(default_value, dict):
            return self._create_dict_widget(default_value)

        # Handle Int
        if arg_type == int or isinstance(default_value, int):
            spin = QSpinBox()
            spin.setFixedWidth(40)
            spin.setRange(-999999, 999999)
            spin.setButtonSymbols(QSpinBox.NoButtons) # Save space
            if isinstance(default_value, int):
                spin.setValue(default_value)
            return spin

        # Handle Float
        if arg_type == float or isinstance(default_value, float):
            spin = QDoubleSpinBox()
            spin.setFixedWidth(40)
            spin.setRange(-999999.0, 999999.0)
            spin.setButtonSymbols(QDoubleSpinBox.NoButtons) # Save space
            if isinstance(default_value, (float, int)):
                spin.setValue(float(default_value))
            return spin

        # Handle String
        if arg_type == str or isinstance(default_value, str):
            line_edit = QLineEdit()
            line_edit.setFixedWidth(100)
            if isinstance(default_value, str):
                line_edit.setText(default_value)
            return line_edit

        return QLabel(f"Unknown type: {arg_type}")

    def _create_dict_widget(self, default_value: dict):
        # Container for nested dict
        # Note: FlowLayout might struggle with large nested widgets if they don't have good size hints.
        # For dicts, we probably want a vertical list inside the flow, or a flow inside.
        # Given user said "nested rectangle", let's keep it simple.
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.2);
                border-radius: 5px;
                border: 1px solid #555555;
            }
        """)
        
        # For dict items, we can also use FlowLayout if we want them to wrap inside the dict block
        layout = FlowLayout(container, margin=5, hSpacing=10, vSpacing=5)

        if not default_value:
            layout.addWidget(QLabel("Empty configuration"))
            return container

        for key, val in default_value.items():
            pair_widget = QWidget()
            pair_layout = QHBoxLayout(pair_widget)
            pair_layout.setContentsMargins(0, 0, 0, 0)
            pair_layout.setSpacing(5)
            
            # Label for key
            lbl = QLabel(str(key))
            lbl.setStyleSheet("color: #bbbbbb; font-style: italic; border: none; background: transparent;")
            pair_layout.addWidget(lbl)
            
            # Input for value
            val_type = type(val)
            widget = self._create_input_widget(val_type, val)
            
            pair_layout.addWidget(widget)
            layout.addWidget(pair_widget)

        return container

    def set_selected(self, selected: bool):
        self._selected = selected
        self.update_style()

    def update_style(self):
        border = "border: 2px solid #3399ff;" if self._selected else ""
        self.setStyleSheet(f"""
            FunctionBlock {{
                background-color: rgb(72, 72, 72);
                border-radius: 10px;
                {border}
            }}
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setData('application/x-function-block', b'')
            drag.setMimeData(mime)
            
            # Create a visual representation of the widget being dragged
            pixmap = QPixmap(self.size())
            self.render(pixmap)
            drag.setPixmap(pixmap)
            drag.setHotSpot(e.pos())
            
            drag.exec_(Qt.MoveAction)
