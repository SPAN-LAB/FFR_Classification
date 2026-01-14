# Qt Signals/Slots Migration Summary

## Overview
Successfully migrated the GUI from a callback-based architecture to Qt's Signal/Slot mechanism for cleaner, more maintainable event-driven communication.

## Changes Made

### 1. Manager Class (`manager.py`)

#### Added Qt Inheritance
```python
class Manager(QObject):  # Now inherits from QObject
    def __init__(self):
        super().__init__()  # Initialize QObject
```

#### Defined Signals
```python
# Signals for various events
subjects_loaded = pyqtSignal(str)      # emits: folder_path
pipeline_loaded = pyqtSignal(str)      # emits: file_path
pipeline_created = pyqtSignal(str)     # emits: file_path
pipeline_saved = pyqtSignal()
function_added = pyqtSignal(str, dict) # emits: name, parameters
functions_updated = pyqtSignal()       # emits when functions list changes
state_changed = pyqtSignal()
```

#### Removed Callback Properties
**Before:**
```python
self.subjects_folder_path_updater = None  # callback
self.pipeline_path_updater = None         # callback
```

**After:** Removed entirely - signals replace these

#### Updated Methods to Emit Signals
**Example - load_subjects():**

Before:
```python
self.subjects_folder_path = folder_path
if self.subjects_folder_path_updater is not None:
    self.subjects_folder_path_updater()
```

After:
```python
self.subjects_folder_path = folder_path
self.subjects_loaded.emit(folder_path)
self.state_changed.emit()
```

#### Added New Helper Methods
- `add_function(name, parameters)` - Adds function and emits signals
- `remove_function(index)` - Removes function and emits signals
- `clear_functions()` - Clears all functions and emits signals
- `reorder_functions(old_index, new_index)` - Reorders and emits signals

### 2. TopBarInfo Component (`top_bar_info.py`)

#### Removed Callback Registration
**Before:**
```python
self.manager.subjects_folder_path_updater = self.set_subjects_directory
self.manager.pipeline_path_updater = self.set_pipeline_directory
```

#### Connected to Signals
**After:**
```python
# Connect to manager signals
self.manager.subjects_loaded.connect(self.on_subjects_loaded)
self.manager.pipeline_loaded.connect(self.on_pipeline_loaded)
self.manager.pipeline_created.connect(self.on_pipeline_loaded)
```

#### Renamed Methods to Slots
**Before:**
```python
def set_subjects_directory(self):
    self.subjects_label.setText(f"Subjects Folder: {self.manager.subjects_folder_path}")
```

**After:**
```python
def on_subjects_loaded(self, folder_path: str):
    """Slot that receives the folder path when subjects are loaded"""
    self.subjects_label.setText(f"Subjects Folder: {folder_path}")
```

### 3. FunctionBuilder Component (`function_builder.py`)

#### Connected to Signals
```python
# Connect to manager signals
self.manager.function_added.connect(self.on_functions_updated)
self.manager.functions_updated.connect(self.on_functions_updated)
self.manager.pipeline_loaded.connect(self.on_functions_updated)
self.manager.pipeline_created.connect(self.on_functions_updated)
```

#### Added Slot Method
```python
def on_functions_updated(self, *args):
    """
    Slot that updates the function blocks when functions change.
    Accepts optional arguments from different signals.
    """
    self.populate_functions()
```

#### Updated Drag & Drop
**Before:**
```python
# Directly mutated manager data and manually triggered update
item = self.manager.functions.pop(source_index)
self.manager.functions.insert(real_target_index, item)
self.populate_functions()
```

**After:**
```python
# Use manager method which emits signals automatically
self.manager.reorder_functions(source_index, real_target_index)
```

### 4. BottomBar Component (`bottom_bar.py`)

#### Simplified Function Addition
**Before:**
```python
# Manually appended and triggered update via parent access
self.manager.functions.append((selected_function, {}))
main_window = self.window()
if hasattr(main_window, 'builder'):
    main_window.builder.populate_functions()
```

**After:**
```python
# Use manager method which emits signals automatically
self.manager.add_function(selected_function, {})
```

## Benefits of This Migration

### 1. **Multiple Subscribers**
```python
# Multiple components can listen to the same signal
self.manager.subjects_loaded.connect(top_bar_info.on_subjects_loaded)
self.manager.subjects_loaded.connect(subject_summary.update_counts)
self.manager.subjects_loaded.connect(log_widget.log_action)
```

### 2. **Type Safety**
- Signals are typed: `pyqtSignal(str)`, `pyqtSignal(str, dict)`
- IDEs and linters can catch mismatches

### 3. **Automatic Cleanup**
- Qt automatically disconnects signals when objects are destroyed
- No memory leaks from dangling callback references

### 4. **Loose Coupling**
- Views no longer modify Manager properties
- Manager doesn't know about specific views
- Easy to add/remove components

### 5. **Better Debugging**
- Signal connections are visible in Qt debuggers
- Clear event flow tracking

### 6. **Thread Safety**
- Signals can safely emit across threads
- Qt handles the synchronization

## Usage Examples

### Listening to Manager Events
```python
class MyNewComponent(QWidget):
    def __init__(self, parent, manager: Manager):
        super().__init__(parent)
        self.manager = manager
        
        # Connect to any signals you need
        self.manager.subjects_loaded.connect(self.handle_subjects_loaded)
        self.manager.function_added.connect(self.handle_function_added)
    
    def handle_subjects_loaded(self, folder_path: str):
        print(f"Subjects loaded from: {folder_path}")
        # Update your UI here
    
    def handle_function_added(self, name: str, parameters: dict):
        print(f"Function added: {name} with params: {parameters}")
        # Update your UI here
```

### Emitting Custom Events
If you need to add new events to the Manager:

```python
class Manager(QObject):
    # Add new signal
    error_occurred = pyqtSignal(str)  # emits: error_message
    
    def some_method(self):
        try:
            # ... do work ...
        except Exception as e:
            self.error_occurred.emit(str(e))
```

## Migration Checklist

✅ Manager inherits from QObject  
✅ All signals defined at class level  
✅ Callback properties removed  
✅ All state-changing methods emit appropriate signals  
✅ TopBarInfo uses signal connections  
✅ FunctionBuilder uses signal connections  
✅ BottomBar uses signal connections  
✅ No direct data manipulation from views  
✅ Helper methods added for common operations  
✅ No linter errors  

## Future Enhancements

The signal-based architecture makes it easy to add:

1. **Undo/Redo System**: Listen to `state_changed` signal
2. **Logging Widget**: Connect all signals to a log view
3. **Status Bar**: Show real-time operation status
4. **Progress Indicators**: Track long-running operations
5. **Error Dialogs**: Display errors from `error_occurred` signal
6. **Auto-save**: Trigger saves on `state_changed` signal

## Testing

The signal-based architecture is easier to test:

```python
from PyQt5.QtTest import QSignalSpy

def test_load_subjects():
    manager = Manager()
    spy = QSignalSpy(manager.subjects_loaded)
    
    manager.load_subjects("/path/to/subjects")
    
    assert spy.count() == 1
    assert spy.at(0) == ["/path/to/subjects"]
```

