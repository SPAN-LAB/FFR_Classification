from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QSizePolicy, QFrame
from PyQt5.QtCore import Qt

from ..manager import Manager
from .function_block import FunctionBlock

from ...core.utils.details import *


class FunctionBuilder(QScrollArea):
    def __init__(self, parent, manager: Manager):
        super().__init__(parent)
        self.manager = manager
        
        # Connect to manager signals
        self.manager.function_added.connect(self.on_functions_updated)
        self.manager.functions_updated.connect(self.on_functions_updated)
        self.manager.pipeline_loaded.connect(self.on_functions_updated)
        self.manager.pipeline_created.connect(self.on_functions_updated)
        
        # Make the scroll area expand to fill available space
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFrameShape(QFrame.NoFrame)
        
        # Configure scroll area with native scroll bars
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Ensure native macOS scroll bars are used
        self.verticalScrollBar().setStyleSheet("")  # Empty stylesheet = use native style
        
        # Create container widget
        self.container = QWidget()
        self.container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Create vertical layout for stacking function blocks
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(8)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Set the container as the scroll area's widget
        self.setWidget(self.container)
        self.setAcceptDrops(True)

        # Track selected block
        self.selected_block = None

        # Initialize drop indicator
        self.drop_indicator = QFrame()
        self.drop_indicator.setFrameShape(QFrame.HLine)
        self.drop_indicator.setLineWidth(2)
        self.drop_indicator.setStyleSheet("color: #3399ff; background-color: #3399ff; min-height: 2px;")
        self.drop_indicator.setFixedHeight(2)
        self.drop_indicator.hide()

        # Initialize function details list (Dummy data for now)
        if not self.manager.functions:
             self.manager.functions = [
                ("Subaverage Trials", {}),
                # ("Split into Folds", {}) 
            ]

        # Populate with function blocks
        self.populate_functions()
    
    def on_functions_updated(self, *args):
        """
        Slot that updates the function blocks when functions change.
        Accepts optional arguments from different signals (like folder_path from pipeline_loaded).
        """
        self.populate_functions()
        
    def on_block_clicked(self, block):
        if self.selected_block and self.selected_block != block:
            self.selected_block.set_selected(False)
        
        self.selected_block = block
        self.selected_block.set_selected(True)

    def populate_functions(self):
        """
        Create FunctionBlocks from self.manager.functions
        """
        # Clear existing items from layout
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Reset selected block since we are rebuilding
        self.selected_block = None

        available_functions = self.manager.find_functions()
        
        for (function_name, parameters) in self.manager.functions:
            if function_name in available_functions:
                function_detail = available_functions[function_name].detail
                function_block = FunctionBlock(parent=self, function_detail=function_detail)
                function_block.clicked.connect(self.on_block_clicked)
                self.layout.addWidget(function_block)
            
        # Add stretch at bottom
        self.layout.addStretch()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat('application/x-function-block'):
            event.accept()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.drop_indicator.setParent(None)
        self.drop_indicator.hide()
        event.accept()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat('application/x-function-block'):
            # Remove indicator temporarily to calculate correct positions
            self.drop_indicator.setParent(None)
            
            pos = event.pos()
            container_pos = self.container.mapFrom(self, pos)
            
            # Find insertion index
            index = 0
            for i in range(self.layout.count()):
                w = self.layout.itemAt(i).widget()
                if not w: continue # skip spacers or invalid items
                
                if container_pos.y() > w.geometry().y() + w.geometry().height() / 2:
                    index = i + 1
            
            # Insert indicator at the calculated position
            # Note: inserting widget changes indices, but for visual feedback it's fine
            self.layout.insertWidget(index, self.drop_indicator)
            self.drop_indicator.show()
            
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasFormat('application/x-function-block'):
            # Find where the indicator is to know the target index
            target_index = self.layout.indexOf(self.drop_indicator)
            
            # Clean up indicator
            self.drop_indicator.setParent(None)
            self.drop_indicator.hide()
            
            if target_index == -1:
                event.ignore()
                return

            source_widget = event.source()
            if isinstance(source_widget, FunctionBlock):
                source_index = self.layout.indexOf(source_widget)
                
                if source_index != -1:
                    # Calculate real target index in the list
                    # If source is before target, the source removal shifts indices
                    real_target_index = target_index
                    if source_index < target_index:
                        real_target_index -= 1
                    
                    # Update data model using the manager method
                    self.manager.reorder_functions(source_index, real_target_index)
                    
                    event.setDropAction(Qt.MoveAction)
                    event.accept()
                    return

        event.ignore()

