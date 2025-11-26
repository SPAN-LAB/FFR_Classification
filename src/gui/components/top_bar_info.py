from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QPalette

from ..manager import Manager


class TopBarInfo(QWidget):
    def __init__(self, parent, manager: Manager):
        super().__init__(parent)
        self.manager = manager
        
        # Connect to manager signals
        self.manager.subjects_loaded.connect(self.on_subjects_loaded)
        self.manager.pipeline_loaded.connect(self.on_pipeline_loaded)
        self.manager.pipeline_created.connect(self.on_pipeline_loaded)
        
        self.setFixedHeight(60)
        
        # Flag to prevent recursion when updating background
        self._updating_background = False
        
        # Set background color that adapts to light/dark mode
        self._update_background_color()
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create labels (left aligned)
        self.subjects_label = QLabel(self.subjects_folder_path_text)
        self.subjects_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.subjects_label.setContentsMargins(10, 5, 10, 5)
        self.pipeline_label = QLabel(self.pipeline_path_text)
        self.pipeline_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.pipeline_label.setContentsMargins(10, 5, 10, 5)
        
        layout.addWidget(self.subjects_label)
        layout.addWidget(self.pipeline_label)
    
    def changeEvent(self, event: QEvent):
        """Handle system theme changes"""
        if event.type() == QEvent.Type.PaletteChange:
            self._update_background_color()
        super().changeEvent(event)
    
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
                # Darker grey for dark mode
                self.setStyleSheet("background-color: rgb(36, 36, 36);")
            else:
                # Lighter grey for light mode
                self.setStyleSheet("background-color: rgb(210, 210, 210);")
        finally:
            self._updating_background = False

    # MARK: View calculations 
    
    @property
    def subjects_folder_path_text(self) -> str:
        if self.manager.subjects_folder_path is not None:
            return self.manager.subjects_folder_path
        return "Subjects Folder: <empty>"
    
    @property
    def pipeline_path_text(self) -> str:
        if self.manager.pipeline_path is not None:
            return self.manager.pipeline_path
        return "Pipeline File: <empty>"
    
    # Slots for updating the UI
    
    def on_subjects_loaded(self, folder_path: str):
        """
        Slot that receives the folder path when subjects are loaded
        """
        self.subjects_label.setText(f"Subjects Folder: {folder_path}")
    
    def on_pipeline_loaded(self, pipeline_path: str):
        """
        Slot that receives the pipeline path when pipeline is loaded or created
        """
        self.pipeline_label.setText(f"Pipeline File: {pipeline_path}")

