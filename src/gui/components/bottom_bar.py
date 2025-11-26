from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QMenu, QAction, QDialog, QListWidget, QVBoxLayout, QDialogButtonBox
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QPalette, QColor

from ..manager import Manager


class FunctionSelectionDialog(QDialog):
    def __init__(self, available_functions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Function")
        self.resize(300, 400)
        
        self.layout = QVBoxLayout(self)
        
        self.list_widget = QListWidget()
        self.list_widget.addItems(sorted(available_functions.keys()))
        self.layout.addWidget(self.list_widget)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)
        
    def get_selected_function(self):
        items = self.list_widget.selectedItems()
        if items:
            return items[0].text()
        return None

class BottomBar(QWidget):
    def __init__(self, parent, manager: Manager):
        super().__init__(parent)
        self.manager = manager
        
        self.setFixedHeight(40)

        self.setAutoFillBackground(True)

        # Flag to prevent recursion when updating background
        self._updating_background = False
        
        # Set background color that adapts to light/dark mode
        self._update_background_color()
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create labels (left aligned)
        self.placeholder1 = QLabel("Placeholder 1", parent=self)
        self.placeholder1.setContentsMargins(10, 5, 10, 5)
        
        self.add_function_btn = QPushButton("Add Function", parent=self)
        self.add_function_btn.clicked.connect(self.show_add_function_dialog)
        
        layout.addWidget(self.placeholder1)
        layout.addWidget(self.add_function_btn)
        layout.addStretch()

    def show_add_function_dialog(self):
        available_functions = self.manager.find_functions()
        dialog = FunctionSelectionDialog(available_functions, self)
        
        if dialog.exec_() == QDialog.Accepted:
            selected_function = dialog.get_selected_function()
            if selected_function:
                # Use manager method which will emit signals automatically
                # Assuming default empty dict for parameters for now
                self.manager.add_function(selected_function, {})

    # def changeEvent(self, event: QEvent):
    #     """Handle system theme changes"""
    #     if event.type() == QEvent.Type.PaletteChange:
    #         self._update_background_color()
    #     super().changeEvent(event)
    
    def _update_background_color(self):
        """Update background color based on current theme"""
        # Prevent recursion
        if self._updating_background:
            return

        self._updating_background = True
        try:
            palette = self.palette()
            
            # Detect if we're in dark mode by checking text color brightness
            # (text is light in dark mode, dark in light mode)
            text_color = palette.color(QPalette.ColorRole.WindowText)
            is_dark_mode = text_color.lightness() > 128
            
            # Set appropriate grey for the mode using stylesheet
            if is_dark_mode:
                pal = self.palette()
                pal.setColor(QPalette.Window, QColor(36, 36, 36))
                self.setPalette(pal)
            else:
                pal = self.palette()
                pal.setColor(QPalette.Window, QColor(210, 210, 210))
                self.setPalette(pal)
        finally:
            self._updating_background = False
    
    