# FFR_Classification

## Overview 

This repo provides convenient methods for ingesting, manipulating, and analyzing EEG data; it operates on instances of `EEGSubject`, about which more detail can be found in the **Project Structure** section. For ease-of-use, run functions through the GUI. (An easy-to-use code implementation will be completed soon, much of the functionality is present.) 

## Why use it

EEG data is manipulated in several frequently-occuring ways before being used for analysis or training ML models. Our objectives were to 
1. Minimize the re-writing of common pre-training algorithms.
2. Enable developers to focus on model architecture by creating an interface between the data and model. 
3. Create an efficient pipeline for pre-processing raw data and creating ML models out of them. 

We have created `EEGTrial` and `EEGSubject` data structures to model data. To use the suite of common pre-training algorithms, simply use the methods of `EEGSubject`, such as `.subverage()` and `.trim_by_timestamp()`. The `Plotter` class provides different ways of visualizing subject data and interfaces well with EEGSubject. For example, we can plot the averages trials from the EEG data of a single subject like so: 

```
# subject is of type EEGSubject 
Plotter.plot_averaged_trials(subject.trials)
```

The `Trainer` class trains a `PyTorch` model on a single subject's data, and has been designed to be as easy to use as so: 

```
# trainer is of type Trainer
trainer.train(use_gpu=True, num_epochs=10, lr=0.001, stopping_criteria=False)
```

## How to use (GUI)

First, create your virtual environment and activate it. 

```
python -m venv <virtual_environment_name>
source <virtual_environment_name>/bin/activate
```

Install the required packages.

```
pip install -r requirements.txt
```

Run the GUI like so:

`python -m main`

