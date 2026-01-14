from PyQt5.QtWidgets import QToolBar, QAction, QFileDialog
import os

from ..manager import Manager

class Toolbar(QToolBar):
    def __init__(self, title: str, parent, manager: Manager):
        super().__init__(title, parent)
        self.manager = manager
        
        # Allow the toolbar to be moved
        self.setMovable(True)

        # Actions 
        load_subjects_action = QAction("Load Subjects", self)
        load_subjects_action.triggered.connect(self.do_load_subjects)
        load_pipeline_action = QAction("Load Pipeline", self)
        load_pipeline_action.triggered.connect(self.do_load_pipeline)
        create_pipeline_action = QAction("Create Pipeline", self)
        create_pipeline_action.triggered.connect(self.do_create_pipeline)

        # Add to the UI
        self.addAction(load_subjects_action)
        self.addSeparator()
        self.addAction(load_pipeline_action)
        self.addSeparator()
        self.addAction(create_pipeline_action)
        self.addSeparator()

        # Layout 

    
    # MARK: Functions

    def do_load_subjects(self):
        # Get the folder path
        path = QFileDialog.getExistingDirectory(
            parent=self, 
            caption="Select Subject Directory", 
            directory=""
        )

        user_cancelled = path == ""

        if not user_cancelled:
            # Ensure we have an absolute path
            absolute_path = os.path.abspath(path)
            self.manager.load_subjects(absolute_path)
    
    def do_load_pipeline(self):
        # Get the file path (single .pkl file only)
        path, _ = QFileDialog.getOpenFileName(
            parent=self, 
            caption="Select Pipeline File", 
            directory="",
            filter="Pickle Files (*.pkl)"
        )

        user_cancelled = path == ""

        if not user_cancelled:
            # Ensure we have an absolute path
            absolute_path = os.path.abspath(path)
            self.manager.load_pipeline(absolute_path)
        
    def do_create_pipeline(self):
        """
        Displays a native file dialog for creating a new .pkl file. 
        Allows the user to set the filename before saving.
        """
        path, _ = QFileDialog.getSaveFileName(
            parent=self,
            caption="Create New Pipeline File",
            directory="",
            filter="Pickle Files (*.pkl)"
        )
        
        user_cancelled = path == ""
        
        if not user_cancelled:
            # Ensure we have an absolute path
            absolute_path = os.path.abspath(path)
            
            # Ensure .pkl extension
            if not absolute_path.endswith('.pkl'):
                absolute_path += '.pkl'
            
            self.manager.create_pipeline(absolute_path)