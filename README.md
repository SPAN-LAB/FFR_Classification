# FFR_Classification

## Overview

`FFR_Classification` is a toolbox for analyzing, visualizing, and training machine learning models on frequency-following response (FFR) EEG data using just a few lines in Python.

## Installation

**Note: This toolbox uses Python 3.11.**

[MacOS](#macos-and-linux)
[Linux](#macos-and-linux)

### MacOS and Linux

First, clone the repository onto your machine:

```bash
git clone https://github.com/SPAN-LAB/FFR_Classification.git
cd FFR_Classification
```

Create a virtual environment and activate it:

```bash
python3.11 -m venv .env
source .env/bin/activate
```

Install dependencies: 

```bash
pip install -r requirements.txt
```

## Example and Walkthrough

Create an `AnalysisPipeline` instance. This object is how we’ll do our operations: 

```python
my_pipeline = AnalysisPipeline()
```

Next, we load our subject data. You can either provide a path to a `.mat` file or to a directory containing multiple `.mat` files. In the latter case, any and all `.mat` files get loaded. 

```python
# Individual file
my_pipeline = my_pipeline.load_subjects("data/S01.mat")
# A whole directory 
my_pipeline = my_pipeline.load_subjects("data")
```

With our data now loaded, we can perform transformations to it: 

```python
my_pipeline = my_pipeline.trim_by_timestamp(start_time=50, end_time=250)
my_pipeline = my_pipeline.subaverage(5)
my_pipeline = my_pipeline.fold(5)
```

You can evaluate the accuracy of an ML model on the loaded data like so: 

```python 
my_pipeline = my_pipeline.evaluate(
  model_name="FFNN",
  training_options={
    "num_epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.1
  }
)
```

Notice that the example code above uses `my_pipeline` many times. We can accomplish the same task much more succinctly using the following syntax: 

```python
only_subjects = BlankPipeline()

my_pipeline = (
  AnalysisPipeline()
  .load_subjects("data")
  .trim_by_timestamp(start_time=50, end_time=250)
  .subaverage(5)
  .fold(5)
  .evaluate(
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

Note two things:

1. The instantiation of several variables using `<var> = BlankPipeline()`. `BlankPipeline` is simply an alias of `AnalysisPipeline`; it is used simply to clarify intentions better.
2. The addition of the `save(to=<pipeline_name>` method which stores the state of the pipeline at various points.