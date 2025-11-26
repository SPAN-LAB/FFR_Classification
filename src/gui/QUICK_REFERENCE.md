# Qt Signals/Slots Quick Reference

## For New Component Developers

### Template for New Components

```python
from PyQt5.QtWidgets import QWidget
from ..manager import Manager

class MyNewComponent(QWidget):
    def __init__(self, parent, manager: Manager):
        super().__init__(parent)
        self.manager = manager
        
        # STEP 1: Connect to signals you care about
        self.manager.subjects_loaded.connect(self.on_subjects_loaded)
        self.manager.function_added.connect(self.on_function_added)
        
        # STEP 2: Set up your UI
        self.init_ui()
    
    def init_ui(self):
        # Create your widgets here
        pass
    
    # STEP 3: Define slots (event handlers)
    def on_subjects_loaded(self, folder_path: str):
        """Called when subjects are loaded"""
        # Update your UI here
        pass
    
    def on_function_added(self, name: str, params: dict):
        """Called when a function is added"""
        # Update your UI here
        pass
    
    # STEP 4: User actions call Manager methods
    def on_my_button_clicked(self):
        """When user clicks your button"""
        # Don't modify manager.functions directly!
        # Call manager methods instead:
        self.manager.add_function("MyFunc", {"arg": 123})
```

## Available Signals

| Signal | Arguments | When Emitted |
|--------|-----------|--------------|
| `subjects_loaded` | `str` (folder_path) | After loading subjects |
| `pipeline_loaded` | `str` (file_path) | After loading pipeline |
| `pipeline_created` | `str` (file_path) | After creating new pipeline |
| `pipeline_saved` | none | After saving pipeline |
| `function_added` | `str, dict` (name, params) | After adding function |
| `functions_updated` | none | When function list changes |
| `state_changed` | none | After any state change |

## Manager Methods (Use These!)

### Reading Data
```python
# Access state (read-only)
manager.subjects_folder_path
manager.pipeline_path
manager.functions  # list of (name, params) tuples
manager.state      # PipelineState object

# Get available functions
available = manager.find_functions()  # dict[str, Callable]
```

### Modifying Data (Emits Signals)
```python
# Load/save
manager.load_subjects(folder_path)
manager.load_pipeline(file_path)
manager.create_pipeline(file_path)
manager.save_pipeline()

# Manipulate functions
manager.add_function(name, parameters)
manager.remove_function(index)
manager.clear_functions()
manager.reorder_functions(old_index, new_index)

# Run functions
manager.run_function(name, **parameters)
manager.run_all_functions()
```

## Common Patterns

### Pattern 1: Update UI When Data Changes
```python
def __init__(self, parent, manager):
    super().__init__(parent)
    self.manager = manager
    
    # Connect signal to update method
    manager.subjects_loaded.connect(self.refresh_ui)
    manager.functions_updated.connect(self.refresh_ui)
    
def refresh_ui(self, *args):
    """Updates entire UI. *args handles different signals"""
    # Read from manager and update widgets
    self.label.setText(f"Functions: {len(self.manager.functions)}")
```

### Pattern 2: Respond to Specific Events
```python
def __init__(self, parent, manager):
    super().__init__(parent)
    manager.function_added.connect(self.on_function_added)

def on_function_added(self, name: str, params: dict):
    """Only called when function is added"""
    self.log(f"Added: {name}")
```

### Pattern 3: User Action → Manager Method
```python
def on_button_clicked(self):
    # ❌ WRONG: Direct modification
    # self.manager.functions.append(("MyFunc", {}))
    
    # ✅ RIGHT: Use manager method
    self.manager.add_function("MyFunc", {})
    # Manager will emit signals, all views update automatically
```

### Pattern 4: Connect Multiple Signals to One Slot
```python
def __init__(self, parent, manager):
    super().__init__(parent)
    
    # All these trigger the same update
    manager.pipeline_loaded.connect(self.update_display)
    manager.pipeline_created.connect(self.update_display)
    manager.functions_updated.connect(self.update_display)

def update_display(self, *args):
    """Handles updates from multiple sources"""
    # Refresh your view
    pass
```

## DO's and DON'Ts

### ✅ DO
```python
# Use manager methods
manager.add_function("Trim", {"start": 0, "end": 100})

# Connect in __init__
manager.subjects_loaded.connect(self.on_subjects_loaded)

# Accept signal arguments
def on_subjects_loaded(self, folder_path: str):
    self.label.setText(folder_path)

# Use *args for flexible slots
def on_update(self, *args):
    self.refresh()
```

### ❌ DON'T
```python
# Don't modify manager data directly
manager.functions.append(("Trim", {}))  # ❌ No signals emitted!

# Don't call other component methods
other_component.update_display()  # ❌ Tight coupling!

# Don't store duplicate state
self.my_functions = manager.functions.copy()  # ❌ Out of sync!

# Don't forget signal arguments
def on_subjects_loaded(self):  # ❌ Missing folder_path argument!
    pass
```

## Debugging

### Check Signal Connections
```python
# Print number of connections
print(manager.subjects_loaded.receivers(manager.subjects_loaded))
```

### Add Logging to Slots
```python
def on_subjects_loaded(self, folder_path: str):
    print(f"[DEBUG] on_subjects_loaded called with: {folder_path}")
    # ... rest of method
```

### Verify Signal Emission
```python
# In Manager method
def load_subjects(self, folder_path: str):
    # ...
    print(f"[DEBUG] Emitting subjects_loaded: {folder_path}")
    self.subjects_loaded.emit(folder_path)
```

## Testing Your Component

```python
from PyQt5.QtTest import QSignalSpy

def test_my_component():
    manager = Manager()
    component = MyComponent(None, manager)
    
    # Test that component responds to signal
    manager.subjects_loaded.emit("/test/path")
    
    # Assert UI updated
    assert "/test/path" in component.label.text()
```

## Need Help?

1. **Read**: `ARCHITECTURE.md` for overall design
2. **Read**: `SIGNALS_MIGRATION.md` for migration details
3. **Check**: Existing components for examples
4. **Search**: Qt documentation for `QObject`, `pyqtSignal`, `connect()`

## Summary

**Remember the golden rule:**

> Views observe Manager through signals.  
> Views modify Manager through methods.  
> Views never talk to each other directly.

This keeps your code clean, maintainable, and testable! 🎉

