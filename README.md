# FFR Classification Toolbox

A Python toolbox for analyzing, classifying, and visualizing frequency-following response (FFR) EEG data using machine learning. Built at the [Speech Processing and Auditory Neuroscience (SPAN) Lab](https://span.waisman.wisc.edu) at the University of Wisconsin–Madison.

---

## What Does This Toolbox Do?

Frequency-following responses (FFRs) are brain signals recorded from the scalp that track the pitch of sounds — like speech tones — in real time. Researchers use FFRs to study how the brain processes language, music, and sound, and to detect hearing or language disorders.

This toolbox provides:

- **A complete pipeline** for loading, preprocessing, and classifying FFR EEG data stored in `.mat` files
- **Multiple machine learning models** including LDA, SVM, CNN, LSTM, and multi-feature neural networks
- **Feature extraction** methods including pitch tracking, autocorrelation, STFT spectrogram, and a learned autoencoder representation
- **A graphical user interface (GUI)** for running the full pipeline without writing any code
- **Visualization tools** including confusion matrices, ROC curves, and averaged EEG waveforms per tone

---

## Contents

- [Installation — Mac and Linux](#installation--mac-and-linux)
- [Installation — Windows](#installation--windows)
- [Running the GUI](#running-the-gui)
- [Running a Script](#running-a-script)
- [Quick Example](#quick-example)
- [Pipeline Functions](#pipeline-functions)
- [Available Models](#available-models)
- [Available Features](#available-features)
- [Troubleshooting](#troubleshooting)

---

## Installation — Mac and Linux

### Step 1 — Install Python 3.11

Check if you already have it:
```bash
python3.11 --version
```
If not:
- **Mac:** Install [Homebrew](https://brew.sh) first, then run `brew install python@3.11`
- **Linux (Ubuntu/Debian):** Run `sudo apt install python3.11 python3.11-venv`

### Step 2 — Install Git

Check if you have it:
```bash
git --version
```
If not: `brew install git` (Mac) or `sudo apt install git` (Linux)

### Step 3 — Clone the repository

```bash
git clone https://github.com/SPAN-LAB/FFR_Classification.git
cd FFR_Classification
```

### Step 4 — Set up the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Every time you open a new terminal**, activate the environment first:
> ```bash
> source .venv/bin/activate
> ```

---

## Installation — Windows

### Step 1 — Install Python 3.11
- Go to https://www.python.org/downloads/release/python-3119/
- Download **Windows installer (64-bit)**
- Run installer — **check "Add Python to PATH"** before clicking Install

### Step 2 — Install Git
- Go to https://git-scm.com/download/win
- Download and install with default settings

### Step 3 — Clone the repository

Open **Command Prompt** (`cmd`) and run:
```cmd
cd %USERPROFILE%
mkdir ffr
cd ffr
git clone https://github.com/SPAN-LAB/FFR_Classification.git
cd FFR_Classification
```

### Step 4 — Set up the environment

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

> **Every time you open a new Command Prompt**, activate the environment first:
> ```cmd
> .venv\Scripts\activate.bat
> ```

> **If you see a PowerShell error about scripts being disabled**, switch to Command Prompt (not PowerShell) and use `.venv\Scripts\activate.bat` instead.

---

## Running the GUI

The GUI lets you load data, build a pipeline, run models, and view results — without writing any code.

**Mac / Linux:**
```bash
python run_gui.py
```

**Windows:**
```cmd
python run_gui.py
```

### How to use the GUI

1. Click **+ Load Subject File** to load a single `.mat` file, or **+ Load Subject Folder** to load a whole folder
2. Click **Add Function** to add steps to your pipeline
3. Click **Edit** on each function to set its parameters
4. Click **Run Functions** to run the pipeline
5. Click on a subject name in the bottom-left list to see results

**Recommended pipeline order:**

| Step | Function | Recommended Parameters |
|------|----------|----------------------|
| 1 | Trim by Timestamp | Start: 50 ms, End: 250 ms |
| 2 | Subaverage Trials | Size: 5 |
| 3 | Split into Folds | Folds: 5 |
| 4 | Extract Features | Select from list |
| 5 | Evaluate Model | Select a model |

---

## Running a Script

If you are comfortable with Python, you can run analyses from a `.py` file.

**Mac / Linux:**
```bash
python your_script.py
```

**Windows:**
```cmd
python your_script.py
```

---

## Quick Example

Here is a complete example that loads FFR data from a single subject, preprocesses it, extracts features, and evaluates a model using 5-fold cross-validation:

```python
from src.core import AnalysisPipeline, PipelineState

# Run the full pipeline on one subject
pipeline = (
    AnalysisPipeline()
    .load_subjects("data/4T1015.mat")        # Load a .mat file
    .trim_by_timestamp(start_time=50,         # Keep only 50–250 ms
                       end_time=250)
    .subaverage(size=5)                       # Average every 5 trials
    .fold(num_folds=5)                        # Split into 5 folds
    .extract_features("pitchtrack,autocorr")  # Extract features
    .evaluate_model(                          # Train and evaluate
        model_name="DynamicFFNN",
        training_options={
            "num_epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.1,
            "patience": 20,
            "min_delta": 0.001,
            "embed_dim": 64,
        }
    )
)
```

You can also load a whole folder of subjects and save snapshots of the pipeline at different stages:

```python
from src.core import AnalysisPipeline, PipelineState

# Snapshot variables to save pipeline state at different points
after_loading = PipelineState()
after_preprocessing = PipelineState()

pipeline = (
    AnalysisPipeline()
    .load_subjects("data/")                   # Load all .mat files in folder
    .save(to=after_loading)                   # Save snapshot after loading
    .trim_by_timestamp(start_time=50, end_time=250)
    .subaverage(size=5)
    .fold(num_folds=5)
    .save(to=after_preprocessing)             # Save snapshot after preprocessing
    .extract_features("pitchtrack,autocorr,autoencoder_latent")
    .evaluate_model(
        model_name="DynamicCNN",
        training_options={
            "num_epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.1,
            "patience": 20,
            "min_delta": 0.001,
            "embed_dim": 64,
        }
    )
)
```

---

## Pipeline Functions

| Function | Description |
|----------|-------------|
| `trim_by_timestamp(start_time, end_time)` | Keep only data within a time window (ms) |
| `trim_by_index` | Keep only certain window of trials |
| `subaverage(size)` | Average every N trials together to improve signal-to-noise ratio |
| `split_into_folds` | Split trials into N folds for cross-validation |
| `extract_features(feature_names)` | Extract features from trials (comma-separated list) |
| `evaluate_model(model_name, training_options)` | Train and evaluate a classifier using cross-validation |
| `train_model(model_name, hyperparameters, output_dirpath)` | Train a model and save it to disk |
| `infer_on_model` | Run a pre-trained model |
| `map_labels(rule_csv)` | Remap tone labels using a CSV rule file |

---

## Available Models

| Model Name | Type | Notes |
|------------|------|-------|
| `LDA` | Linear Discriminant Analysis | Fast, good baseline |
| `SVM` | Support Vector Machine | Good for small datasets |
| `FFNN` | Feed-Forward Neural Network | Standard neural network |
| `CNN` | Convolutional Neural Network | Good for raw waveforms |
| `LSTM` | Long Short-Term Memory | Captures temporal patterns |
| `RNN` | Recurrent Neural Network | |
| `GRU` | Gated Recurrent Unit | |
| `Transformer` | Transformer | |
| `DynamicFFNN` | Multi-branch FFNN | Combines multiple feature types |
| `DynamicCNN` | Multi-branch CNN | Best overall accuracy |
| `MultiInputTransformer` | A Multi-Input Transformer | Takes different features as inputs |
| `DynamicCRNN` | A Multi Input RNN model with a CNN base layer |


---

## Available Features

| Feature Name | Description | Output Size |
|--------------|-------------|-------------|
| `raw` | Raw trimmed EEG waveform | 3,277 points |
| `pitchtrack` | Sliding-window pitch track (F0 over time) | 272 points |
| `autocorr` | Full-signal autocorrelation | 3,277 points |
| `spectrogram` | Sliding-window STFT magnitude spectrogram | Variable (flattened) |
| `autoencoder_latent` | 128-dim learned representation from trained autoencoder | 128 points |

---

## Troubleshooting

**"No module named PyQt5"**
You forgot to activate the virtual environment. Run `source .venv/bin/activate` (Mac/Linux) or `.venv\Scripts\activate.bat` (Windows).

**"index out of bounds" error when loading a `.mat` file**
Make sure your `.mat` file contains the fields `ffr_nodss`, `labels`, and `time`. Files from other datasets may have a different structure.

**The GUI opens but shows no subjects after running**
After running the pipeline, click on the subject name in the bottom-left panel to load the results.

**Pipeline gives wrong results on a second run without restarting**
Click **Run Functions** again — the pipeline automatically resets to the original loaded data before each run.

**PyTorch is slow on Windows**
Windows does not support Apple MPS. The toolbox will automatically fall back to CPU. For faster training, use a Mac with Apple Silicon or a Linux machine with a GPU.

---

## Lab

Developed at the [Speech Processing and Auditory Neuroscience (SPAN) Lab](https://span.waisman.wisc.edu), University of Wisconsin–Madison.