# FFR Classification

## Overview

FFR Classification is a toolbox for analyzing, visualizing, and training machine learning models on frequency-following response (FFR) EEG data. It comes with both a graphical user interface (GUI) for point-and-click use and a Python scripting interface for more advanced workflows.

---

## For Mac and Linux

### Step 1 — Install the required software

You will need two things installed on your computer before you can use this toolbox:

**1. Python 3.11**

Check if you already have it by opening Terminal and typing:
```bash
python3.11 --version
```
If you see something like `Python 3.11.x`, you already have it. If not:
- **Mac:** Install [Homebrew](https://brew.sh) first (copy and paste the command from their website into Terminal), then run:
  ```bash
  brew install python@3.11
  ```
- **Linux:** Run:
  ```bash
  sudo apt install python3.11 python3.11-venv
  ```

**2. Git**

Check if you have it:
```bash
git --version
```
If not:
- **Mac:** Run `brew install git`
- **Linux:** Run `sudo apt install git`

---

### Step 2 — Download the code

Open Terminal and run these commands one by one:

```bash
git clone https://github.com/SPAN-LAB/FFR_Classification.git
cd FFR_Classification
git checkout cj_dev
```

This downloads the code onto your computer and puts you in the right folder.

---

### Step 3 — Set up the environment

Think of this as creating a clean workspace just for this project so it doesn't interfere with anything else on your computer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Every time you open a new Terminal window, you need to activate the environment again before running anything:
> ```bash
> cd FFR_Classification
> source .venv/bin/activate
> ```

---

### Running the GUI (Mac / Linux)

The GUI lets you load EEG data, build a pipeline, run models, and see results — all without writing any code.

```bash
python run_gui.py
```

A window will open. Here is how to use it:

1. Click **+ Load Subject File** to load a single `.mat` file, or **+ Load Subject Folder** to load a whole folder of `.mat` files.
2. Click **Add Function** to add steps to your pipeline (e.g. Trim by Timestamp, Subaverage, Split into Folds, Extract Features, Evaluate Model).
3. For each function you add, click **Edit** to set its parameters.
4. Click **Run Functions** to run the full pipeline.
5. Once finished, click on a subject name in the bottom-left list to see the confusion matrix, ROC curve, and averaged EEG waveforms.

**Recommended pipeline order:**
1. Trim by Timestamp (Start: 50 ms, End: 250 ms)
2. Subaverage Trials (size: 5)
3. Split into Folds (folds: 5)
4. Extract Features (select from the list)
5. Evaluate Model (select a model)

---

### Running the Script (Mac / Linux)

If you are comfortable with Python, you can run analyses directly from a script. Here is a basic example:

```python
from src.core import AnalysisPipeline, PipelineState

my_pipeline = AnalysisPipeline()

my_pipeline = (
    my_pipeline
    .load_subjects("data/S01.mat")   # or a folder: .load_subjects("data/")
    .trim_by_timestamp(start_time=50, end_time=250)
    .subaverage(5)
    .fold(5)
    .evaluate_model(
        model_name="FFNN",
        training_options={
            "num_epochs": 20,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.1
        }
    )
)
```

You can also save snapshots of the pipeline at different stages:

```python
from src.core import AnalysisPipeline, PipelineState

only_subjects = PipelineState()
after_transforms = PipelineState()

my_pipeline = (
    AnalysisPipeline()
    .load_subjects("data/")
    .save(to=only_subjects)              # snapshot after loading
    .trim_by_timestamp(start_time=50, end_time=250)
    .subaverage(5)
    .fold(5)
    .save(to=after_transforms)           # snapshot after preprocessing
    .evaluate_model(
        model_name="FFNN",
        training_options={
            "num_epochs": 20,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.1
        }
    )
)
```

Run your script from Terminal (make sure the environment is activated first):

```bash
python your_script.py
```

---

---

## For Windows

### Step 1 — Install the required software

**1. Python 3.11**

- Go to [https://www.python.org/downloads/release/python-3119/](https://www.python.org/downloads/release/python-3119/)
- Download **Windows installer (64-bit)**
- Run the installer — **important:** check the box that says **"Add Python to PATH"** before clicking Install

**2. Git**

- Go to [https://git-scm.com/download/win](https://git-scm.com/download/win)
- Download and run the installer, keeping all the default settings

---

### Step 2 — Download the code

Open **Command Prompt** (search for `cmd` in the Start menu) and run:

```cmd
cd %USERPROFILE%
mkdir ffr
cd ffr
git clone https://github.com/SPAN-LAB/FFR_Classification.git
cd FFR_Classification
git checkout cj_dev
```

---

### Step 3 — Set up the environment

Still in Command Prompt, run these one by one:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Every time you open a new Command Prompt window, you need to activate the environment again:
> ```cmd
> cd %USERPROFILE%\ffr\FFR_Classification
> .venv\Scripts\activate.bat
> ```

> **If you see a PowerShell error about scripts being disabled**, switch to Command Prompt (not PowerShell) and use `.venv\Scripts\activate.bat` instead.

---

### Running the GUI (Windows)

```cmd
python run_gui.py
```

A window will open. Follow the same steps as described in the Mac/Linux section above.

---

### Running the Script (Windows)

Create a Python file (e.g. `my_analysis.py`) with your code, then run it from Command Prompt:

```cmd
python my_analysis.py
```

See the Mac/Linux scripting section above for example code — it works the same way on Windows.


## Troubleshooting

**"No module named PyQt5"** — You forgot to activate the virtual environment. Run `source .venv/bin/activate` (Mac/Linux) or `.venv\Scripts\activate.bat` (Windows).

**"index out of bounds" error when loading a .mat file** — Make sure your `.mat` file contains the fields `ffr_nodss`, `labels`, and `time`. Files from other datasets may have a different structure.

**The GUI runs but shows no subjects** — Make sure you clicked **+ Load Subject File** or **+ Load Subject Folder** before running the pipeline.

**Pipeline gives wrong results on second run** — This is fixed in `cj_dev`. Make sure you are on the right branch: `git checkout cj_dev`.