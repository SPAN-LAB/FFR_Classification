# GUI Architecture - Signal/Slot Pattern

## Component Hierarchy

```
MainWindow (QMainWindow)
│
├── Toolbar (QToolBar)
│   └── Actions: Load Subjects, Load Pipeline, Create Pipeline
│
├── TopBarInfo (QWidget)
│   └── Displays: Subjects path, Pipeline path
│
├── Main Content (QSplitter)
│   ├── FunctionBuilder (QScrollArea)
│   │   └── FunctionBlock widgets (draggable, reorderable)
│   └── Right Panel (placeholder)
│
└── BottomBar (QWidget)
    └── Actions: Add Function
```

## Signal Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Manager (QObject)                    │
│                                                              │
│  State:                          Signals:                   │
│  • PipelineState                 • subjects_loaded          │
│  • functions[]                   • pipeline_loaded          │
│  • subjects_folder_path          • pipeline_created         │
│  • pipeline_path                 • pipeline_saved           │
│                                  • function_added           │
│  Methods:                        • functions_updated        │
│  • load_subjects()               • state_changed            │
│  • load_pipeline()                                          │
│  • create_pipeline()                                        │
│  • add_function()                                           │
│  • remove_function()                                        │
│  • reorder_functions()                                      │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           │ SIGNALS            │ SIGNALS            │ SIGNALS
           ▼                    ▼                    ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │  TopBarInfo │      │  Function   │      │  BottomBar  │
    │             │      │  Builder    │      │             │
    │ Slots:      │      │             │      │ Slots:      │
    │ • on_       │      │ Slots:      │      │ (future)    │
    │   subjects_ │      │ • on_       │      │             │
    │   loaded()  │      │   functions_│      │             │
    │ • on_       │      │   updated() │      │             │
    │   pipeline_ │      │             │      │             │
    │   loaded()  │      │             │      │             │
    └─────────────┘      └─────────────┘      └─────────────┘
```

## Event Flow Examples

### Example 1: Loading Subjects

```
User Action
    │
    ▼
Toolbar.do_load_subjects()
    │
    ▼
Manager.load_subjects(folder_path)
    │
    ├── Update internal state
    ├── self.subjects_folder_path = folder_path
    ├── self.state.load_subjects(folder_path)
    │
    ├── Emit: subjects_loaded(folder_path) ────────┐
    └── Emit: state_changed() ─────────────────┐   │
                                               │   │
         ┌─────────────────────────────────────┘   │
         │                                         │
         ▼                                         ▼
    TopBarInfo.on_subjects_loaded(folder_path)   (other listeners)
         │
         ▼
    Update UI Label
```

### Example 2: Adding Function

```
User Action (Click "Add Function")
    │
    ▼
BottomBar.show_add_function_dialog()
    │
    ├── Show dialog
    ├── User selects function
    │
    ▼
Manager.add_function(name, parameters)
    │
    ├── self.functions.append((name, parameters))
    │
    ├── Emit: function_added(name, parameters) ───┐
    ├── Emit: functions_updated() ────────────┐   │
    └── Emit: state_changed() ────────────┐   │   │
                                          │   │   │
    ┌─────────────────────────────────────┘   │   │
    │     ┌───────────────────────────────────┘   │
    │     │     ┌─────────────────────────────────┘
    │     │     │
    ▼     ▼     ▼
FunctionBuilder.on_functions_updated()
    │
    ▼
Rebuild function blocks UI
```

### Example 3: Drag & Drop Reorder

```
User Action (Drag FunctionBlock)
    │
    ▼
FunctionBuilder.dropEvent()
    │
    ├── Calculate old_index and new_index
    │
    ▼
Manager.reorder_functions(old_index, new_index)
    │
    ├── Move item in self.functions[]
    │
    ├── Emit: functions_updated() ────────────┐
    └── Emit: state_changed() ────────────┐   │
                                          │   │
    ┌─────────────────────────────────────┘   │
    │     ┌───────────────────────────────────┘
    │     │
    ▼     ▼
FunctionBuilder.on_functions_updated()
    │
    ▼
Rebuild function blocks UI
    (Now in new order)
```

## Key Design Principles

### 1. Single Source of Truth
- **Manager** holds all state
- Views never store duplicate data
- Views read from Manager, never write directly

### 2. Unidirectional Data Flow
```
User Action → Component → Manager Method → Update State → Emit Signal → Update Views
```

### 3. Separation of Concerns
- **Manager**: Business logic, state management, signal emission
- **Components**: UI rendering, user interaction, signal handling
- **No cross-component communication**: All through Manager

### 4. Loose Coupling
- Components don't know about each other
- Components only know about Manager interface (signals)
- Easy to add/remove components

### 5. Reactive Updates
- Views automatically update when Manager state changes
- No manual view synchronization needed
- Connect to relevant signals, update happens automatically

## Adding New Components

To add a new component that reacts to Manager events:

```python
class MyNewComponent(QWidget):
    def __init__(self, parent, manager: Manager):
        super().__init__(parent)
        self.manager = manager
        
        # 1. Connect to relevant signals
        self.manager.subjects_loaded.connect(self.on_subjects_loaded)
        self.manager.function_added.connect(self.on_function_added)
        self.manager.state_changed.connect(self.on_state_changed)
        
        # 2. Set up your UI
        self.setup_ui()
    
    # 3. Define slot methods
    def on_subjects_loaded(self, folder_path: str):
        # Update your UI based on new subjects
        pass
    
    def on_function_added(self, name: str, params: dict):
        # Update your UI based on new function
        pass
    
    def on_state_changed(self):
        # Refresh your entire view
        pass
    
    # 4. User actions call Manager methods
    def on_button_clicked(self):
        # Don't modify Manager state directly
        # Call Manager methods which will emit signals
        self.manager.add_function("MyFunction", {"param": 123})
```

## Signal Best Practices

### DO ✅
- Connect signals in `__init__`
- Use descriptive slot method names (`on_subjects_loaded` not `update`)
- Let Manager methods emit signals
- Pass relevant data with signals
- Use type hints for slot parameters

### DON'T ❌
- Don't modify Manager state directly from views
- Don't call other component methods directly
- Don't emit signals from views
- Don't store duplicate state in views
- Don't forget to inherit from QObject for signal emitters

## Testing Strategy

### Unit Testing Components
```python
def test_top_bar_info_updates():
    manager = Manager()
    top_bar = TopBarInfo(None, manager)
    
    # Emit signal
    manager.subjects_loaded.emit("/path/to/subjects")
    
    # Assert UI updated
    assert "Subjects Folder: /path/to/subjects" in top_bar.subjects_label.text()
```

### Integration Testing
```python
def test_full_flow():
    manager = Manager()
    top_bar = TopBarInfo(None, manager)
    builder = FunctionBuilder(None, manager)
    
    # Perform action
    manager.load_subjects("/path/to/subjects")
    
    # Assert multiple components updated
    assert top_bar.subjects_label.text() == "Subjects Folder: /path/to/subjects"
    assert manager.subjects_folder_path == "/path/to/subjects"
```

## Future Enhancements

Ready for these additions thanks to signal architecture:

1. **Log Widget**: Connect to all signals, display timestamped log
2. **Status Bar**: Show real-time operation status
3. **Undo/Redo**: Track `state_changed` emissions
4. **Auto-save**: Trigger on `state_changed` with debouncing
5. **Multi-window**: Share same Manager across windows
6. **Remote GUI**: Emit signals across network (advanced)

