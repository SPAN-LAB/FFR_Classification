# Work Done

This file summarizes the implementation work completed on `anu_dev`.

## Branch Context

- Current branch: `anu_dev`
- GUI source referenced from: `origin/Emily_dev`
- I did not merge all of `Emily_dev` because that branch is older than `anu_dev` in the model/feature layer and would remove newer files such as feature extractors and newer models.
- Instead, I selectively brought over the GUI files and logo.

## GUI Updates

Added the GUI implementation from `origin/Emily_dev`:

- `src/GUI/gui.py`
- `src/GUI/manager.py`
- `src/GUI/widgets.py`
- `src/GUI/README.md`
- `src/GUI/__init__.py`
- `spanlab_logo_final.png`

Added a `.gitignore` exception so the GUI logo can be tracked:

- `!spanlab_logo_final.png`

## Checkpoint / Interrupt Saving

Added GUI checkpoint support so interrupted runs can be recovered.

New GUI controls:

- `Save Checkpoint`
- `Load Checkpoint`

Autosave behavior:

- Autosaves to `.ffr_gui_autosave.pkl`
- Autosaves after pipeline edits
- Autosaves after loading subjects
- Autosaves when a pipeline run starts
- Autosaves after each completed function step
- Autosaves on pipeline failure

Checkpoint state includes:

- Current pipeline state
- Initial loaded-subject state
- GUI pipeline function list
- Pending function queue
- Completed step count
- Run log text

## Feature Persistence

Added pipeline methods for saving and loading extracted features:

- `save_features(filepath)`
- `load_features(filepath)`

The saved feature file includes:

- Subject name
- Source filepath
- Trial index
- Raw input data
- Timestamps
- Raw and mapped labels
- Extracted `trial.features`

`load_features()` attaches saved features back onto currently loaded subjects by matching:

- Subject name
- Trial index

## Visualization Data Persistence

Added pipeline methods for saving and loading data needed for later visualization:

- `save_visualization_data(filepath)`
- `load_visualization_data(filepath)`

The visualization file includes:

- Raw trial data
- Timestamps
- Labels
- Extracted features
- Predictions
- Prediction distributions
- Subject label maps

This allows later plotting without rerunning feature extraction or model evaluation.

## Full Pipeline State Persistence

Added full state save/load methods:

- `save_state(filepath)`
- `load_state(filepath)`

These pickle the full pipeline state, including:

- Loaded subjects
- Extracted features
- Predictions
- Trained model objects

## GUI Function Metadata

Updated GUI metadata in `src/core/utils/details.py` so these functions appear in the GUI function list:

- `extract_features`
- `save_state`
- `load_state`
- `save_features`
- `load_features`
- `save_visualization_data`
- `load_visualization_data`

Also updated the GUI parameter editor so it maps form fields to actual function signatures instead of relying on dictionary order.

## Feature Extraction Improvements

Updated `AnalysisPipeline.extract_features()` to accept either:

- A list, e.g. `["pitchtrack", "autocorr"]`
- A comma-separated string, e.g. `"pitchtrack,autocorr"`

Supported features are currently:

- `pitchtrack`
- `autocorr`
- `zerocrossing`

## MultiBranchCNN Feature Selection

Updated `src/models/MultiBranchCNN.py` so the model can choose feature branches through training options.

New training option:

```python
{
    "feature_inputs": "raw,pitchtrack"
}
```

Examples:

```python
{"feature_inputs": "raw,pitchtrack"}
{"feature_inputs": "raw,autocorr"}
{"feature_inputs": "raw,pitchtrack,zerocrossing"}
```

The pipeline now asks models for required inputs using:

```python
required_inputs_for_options(training_options)
```

That lets `evaluate_model()`, `evaluate_generic_model()`, and `train_model()` automatically extract only the features needed by the selected model configuration.

## Model Discovery Robustness

Updated `src/models/utils/resolver.py` so a model with missing optional dependencies does not break discovery for all models.

If a model module fails to import, discovery now logs a warning and skips that module.

This was needed because the current local environment has dependency issues:

- TensorFlow is missing, so `Jason_CNN` cannot import.
- SciPy/sklearn fail because Anaconda is missing `liblapack.3.dylib`, affecting models like `LDA` and `Autoencoder`.

## Existing User Changes Preserved

These files already had uncommitted changes before this work and were preserved:

- `server_analysis/run_eval.py`
- `src/core/analysis_pipeline.py`
- `src/models/utils/torchnn_base.py`

Additional edits were made where needed without reverting the existing changes.

## Validation Run

Compilation check:

```bash
python -m compileall src/core src/models src/features src/GUI
```

GUI import check:

```bash
python -c "from src.GUI.gui import MainWindow; print('gui import ok')"
```

GUI function discovery check:

```bash
python -c "from src.GUI.manager import Manager; print(sorted(Manager().find_functions().keys()))"
```

Feature and visualization persistence smoke test:

```bash
python -c "import numpy as np, tempfile; from pathlib import Path; from src.core import AnalysisPipeline, EEGSubject, EEGTrial; s=EEGSubject(); t=EEGTrial(subject=s,data=np.array([1.,2.,3.]),trial_index=0,timestamps=np.array([0.,1.,2.]),raw_label=1); t.features={'x':np.array([9.])}; t.prediction=1; t.prediction_distribution={1:0.8,2:0.2}; s.trials=[t]; s.source_filepath='synthetic.mat'; s.setup_labels_map(); p=AnalysisPipeline(); p.subjects=[s]; d=Path(tempfile.gettempdir()); p.save_features(str(d/'ffr_features_test.pkl')); t.features={}; p.load_features(str(d/'ffr_features_test.pkl')); assert 'x' in t.features; p.save_visualization_data(str(d/'ffr_viz_test.pkl')); q=AnalysisPipeline(); q.load_visualization_data(str(d/'ffr_viz_test.pkl')); assert q.subjects[0].trials[0].prediction == 1; print('persistence ok')"
```

MultiBranchCNN configurable branch smoke test:

```bash
python -c "import torch; from src.models.MultiBranchCNN import MultiBranchCNNModel; m=MultiBranchCNNModel({'feature_inputs':'raw,pitchtrack,zerocrossing','n_classes':4}); y=m.model({'raw':torch.randn(2,64),'pitchtrack':torch.randn(2,32),'zerocrossing':torch.randn(2,64)}); assert tuple(y.shape)==(2,4); print('multibranch ok')"
```

## How To Run The GUI

From the repo root:

```bash
python -m src.GUI.gui
```

## Current Git Status Notes

At the end of the work, the tree contains modified and untracked files. New GUI files are untracked until added with Git.

Important new/untracked paths:

- `src/GUI/`
- `spanlab_logo_final.png`
- `WORK_DONE.md`

Important modified paths:

- `.gitignore`
- `src/core/analysis_pipeline.py`
- `src/core/utils/details.py`
- `src/models/MultiBranchCNN.py`
- `src/models/utils/model_interface.py`
- `src/models/utils/resolver.py`
- `src/GUI/gui.py`
- `src/GUI/manager.py`
- `src/GUI/widgets.py`

## Follow-up Changes

The following requested low-hanging-fruit changes were completed after the initial GUI/persistence work.

### Map Labels Accepts Any Label Type

Updated `EEGSubject.map_trial_labels()` so it no longer casts labels to `int`.

Mapping CSV tokens are now parsed with `ast.literal_eval()` when possible, and otherwise kept as strings. This supports labels such as:

- `1`
- `1.5`
- `toneA`
- `"tone A"`
- other scalar label values loaded from `.mat` files

The subject label map is refreshed after mapping.

### Multi-file GUI Selection

Updated the GUI subject-file picker from single-file selection to multi-file selection.

The **Load Subject File** button now supports selecting multiple `.mat` files at once with Control/Command/Shift multi-select.

### Dynamic Waveform Plot Count

Updated the GUI waveform plotting panel so it creates one waveform plot per unique raw label instead of using a fixed 2x2 grid.

Important behavior:

- Waveform plots now group by `trial.raw_label`.
- This prevents mapped classification categories from collapsing distinct tones in the waveform view.
- A 16-tone dataset should now produce 16 waveform slots, assuming 16 unique raw labels are loaded.

### Trim By Type

Added a new pipeline function:

```python
trim_by_type(label_values, label_source="raw")
```

Examples:

```python
pipeline.trim_by_type("1,2,3")
pipeline.trim_by_type("toneA,toneB", label_source="raw")
pipeline.trim_by_type("category1", label_source="mapped")
```

This keeps only selected trial categories before classification. The default label source is `raw`, which matches the tone-selection use case.

The function is also exposed in the GUI as **Trim by Type**.

### Environment Name

Updated installation docs to use the environment name:

```bash
classiFFRy
```

Updated files:

- `README.md`
- `src/GUI/README.md`

### Additional Validation

Additional checks run:

```bash
python -m compileall src/core src/models src/features src/GUI
```

```bash
python -c "from src.GUI.manager import Manager; funcs=Manager().find_functions(); print('trim_by_type' in funcs, 'map_labels' in funcs)"
```

```bash
python -c "from src.GUI.gui import MainWindow; print('gui import ok')"
```

```bash
python -c "import numpy as np, tempfile; from pathlib import Path; from src.core import EEGSubject, EEGTrial; s=EEGSubject(); s.trials=[EEGTrial(subject=s,data=np.array([1.]),trial_index=0,timestamps=np.array([0.]),raw_label='toneA'), EEGTrial(subject=s,data=np.array([2.]),trial_index=1,timestamps=np.array([0.]),raw_label='toneB')]; p=Path(tempfile.gettempdir())/'labels_any_type.csv'; p.write_text('cat1,toneA\ncat2,toneB\n'); s.map_trial_labels(str(p)); assert [t.mapped_label for t in s.trials] == ['cat1','cat2']; s.trim_by_type('toneA'); assert len(s.trials)==1 and s.trials[0].raw_label=='toneA'; print('label mapping and trim ok')"
```

```bash
python -c "import numpy as np; from src.core import EEGSubject, EEGTrial; s=EEGSubject(); s.trials=[EEGTrial(subject=s,data=np.array([1.]),trial_index=0,timestamps=np.array([0.]),raw_label=1,mapped_label='A'), EEGTrial(subject=s,data=np.array([2.]),trial_index=1,timestamps=np.array([0.]),raw_label=2,mapped_label='A')]; by_current=s.grouped_trials(); by_raw=s.grouped_trials(key=lambda t:t.raw_label); assert len(by_current)==1 and len(by_raw)==2; print('raw grouping ok')"
```
