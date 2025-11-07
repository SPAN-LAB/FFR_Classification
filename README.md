# FFR_Classification

## Overview

FFR_Classification is a toolbox for training machine-learning models to classify frequency-following response (FFR) EEG data. FFRs are scalp-recorded potentials that phase-lock to periodic auditory stimuli, showing how the brainstem tracks speech and tone contours. Building robust classifiers on these signals is challenging: researchers must align trials, denoise recordings, engineer features, and wire custom models together while keeping experiments reproducible.

This toolbox streamlines that workflow. We make FFR classification accessible to both script-heavy and point-and-click users by packaging reusable data abstractions, preprocessing operators, neural network interfaces, and visualization tools into one pipeline. The same `EEGSubject` object can flow from ingestion to preprocessing, directly into neural network training loops, or into guided analyses inside the GUI.

Highlights:
- Pipeline-first design that ingests raw `.mat` files, keeps transformations chained on the `EEGSubject` API, and feeds the same object into downstream stages.
- A suite of neural preprocessing interfaces (subaveraging, trimming, folding, label mapping) engineered to reduce boilerplate while staying faithful to FFR research conventions.
- Ready-to-train machine-learning models: the `Trainer` dynamically loads CNN, RNN, FFNN and supports custom PyTorch architectures from `src/models`, handling device placement, cross-validation splits, and checkpointing.
- Visualization utilities that tap directly into the live subject object for quick inspection of single trials, label-wise averages, and grand averages.

## Programmatic Use

```python
from src.core.EEGSubject import EEGSubject
from src.core.Plots import Plots
from src.core.Trainer import Trainer

subject = (
    EEGSubject.init_from_filepath("path/to/ffr_subject.mat")
    .trim_by_timestamp(start_time=0.0, end_time=0.6)
    .subaverage(size=5)
)

Plots.plot_averaged_trials(subject)

trainer = Trainer(subject=subject, model_name="CNN")
trainer.train(use_gpu=True, num_epochs=10, lr=1e-3, stopping_criteria=False)
```

Browse `src/core` for additional helpers such as `EEGSubject.grouped_trials(...)`, label-mapping utilities, and richer plotting options.

## GUI Quick Start

1. Create and activate a virtual environment.

```
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```
pip install -r requirements.txt
```

3. Launch the GUI.

```
python main.py
```

The GUI exposes the same preprocessing, visualization, and training functions that are available in code.

## Design

The system is built around a data-first pipeline that keeps `EEGSubject` instances as the single source of truth. Whether you work in Python or through the GUI, the same objects flow from ingestion through preprocessing, visualization, and training.

### Data Model and Transformation Pipeline

- **Trials as primitives.** `EEGTrial` stores the waveform (`data`), timestamps, and raw or mapped labels for a single presentation. Methods such as `trim_by_index(...)` and `trim_by_timestamp(...)` operate in place so downstream stages always see the updated arrays.
- **Subjects as fluent containers.** `EEGSubject.init_from_filepath(...)` reads `.mat` exports and constructs a subject with a list of `EEGTrial` objects plus bookkeeping like the origin file path. Every transformation method (`trim_by_timestamp`, `trim_by_index`, `subaverage`, `fold`, `map_trial_labels`, `grouped_trials`) returns `self`. That enables fluent chains—`subject.map_trial_labels(csv).subaverage(5).trim_by_timestamp(0.0, 0.6)`—without re-instantiating subjects or copying trials.
- **Label-aware grouping.** `EEGSubject.grouped_trials(...)` underpins both subaveraging and k-fold splitting by organizing trials across labels (raw or mapped). Label mapping can be swapped in at any point via `map_trial_labels(...)`, which raises on unmapped labels to keep the dataset in a consistent state.

### Visualization Layer

- **Plots module.** `Plots.plot_single_trial`, `Plots.plot_averaged_trials`, and `Plots.plot_grand_average` consume live `EEGSubject`/`EEGTrial` instances. When `plot_averaged_trials` is called, the helper internally constructs per-label pseudo subjects, runs `subaverage`, and normalizes y-axis scaling so comparisons are meaningful across labels.
- **No serialization step.** Because the plotting helpers operate directly on the mutated subject, your exploratory visuals always reflect the exact preprocessing steps produced by the chained method calls above.

### Training Flow

- **Trainer construction.** `Trainer(subject=..., model_name=...)` binds a specific subject to a model archetype. During training the class resolves devices (`cuda`, `mps`, or CPU), ensures the subject is folded (falling back to a 5-fold split), resets seeds, and flips trials to use raw labels unless you mapped them otherwise.
- **Dynamic model discovery.** Given a `model_name`, `Trainer` loads `src/models/<model_name>.py` at runtime and instantiates the exported `Model` class with the inferred input size. You can add new architectures by dropping a new file into `src/models` without touching trainer code.
- **Cross-validation and loaders.** Helper methods (`create_train_val_dls`, `create_test_dl`) convert the subject’s folds into `TensorDataset`/`DataLoader` pairs. A `StratifiedShuffleSplit` carve-out creates validation splits inside each training fold so the training loop can track accuracy and loss per epoch. Results, checkpoints, and plots are stored in `outputs/train/<subject>/foldX/` (created on demand) while confusion matrices and ROC curves are generated during testing.

### GUI Architecture

- **Automatic function discovery.** On startup `GUIFunctionManager` inspects `src/gui/user_functions.py` for functions decorated with `@function_label` that begin with `GLOBAL_`. Each function is registered under its label, meaning any new preprocessing or training helper becomes available in the GUI immediately.
- **Specifications for rendering.** `Specification.FunctionSpecification` introspects a callable’s signature along with the `@param_labels` metadata to build `ArgumentSpecification` objects (label, name, type hint, default). The Qt interface uses this specification to render the right input widgets and collect arguments without manual UI code.
- **Central orchestration.** `GUIFunctionManager` keeps an ordered array of the selected functions, exposes the available labels for “Add” menus, and executes the pipeline by calling the chosen functions with GUI-supplied kwargs. This mirrors the programmatic flow: GUI actions mutate the shared `EEGSubject` instances stored in `user_functions.SUBJECTS` so plots and training launched from the GUI operate on the same data that preprocessing produced.
