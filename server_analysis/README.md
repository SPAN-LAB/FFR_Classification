# Some instructions

To analyze the effect of subaverage size, data amount, etc. on accuracy, simply create a `.py` file in this directory, copy the contents of `example.py` into it, and **execute the code from the root directory**. 

## Detailed walkthrough

Step 1: Configure `SUBJECT_FILEPATHS`, found in `/src/analysis/config.py`, with the paths to the subject files.

```python
SUBJECT_FILEPATHS = ["Martian001.mat", "Martian051.mat"]
```

Step 2: Create your file in `src/server_analysis` and import the module(s). I'll call it `subaverage_ffnn.py`.

```python
from src.analysis import subaverage_size
from src.analysis import data_amount
```

Step 3: Call the `analyze` function of the appropriate module. 

```python
subaverage_size.analyze("FFNN")
```

Step 4: Run the function from the **ROOT DIRECTORY**. (Don't include the `.py` extension.)

```bash
python -m server_analysis.subaverage_ffnn
```

## CHTC analysis jobs

CHTC analysis submissions use `server_analysis.run_job`. Each queue row runs
one model, subject, and analysis value while keeping all cross-validation folds
inside that process.

Submit from the job configuration directory so its relative paths resolve:

```bash
cd jobs_config/ffr_test_run
mkdir -p logs analyses/subaverage analyses/data_amount
condor_submit subaverage.sub
# or: condor_submit data_amount.sub
```

`subaverage_jobs.txt` and `data_amount_jobs.txt` contain rows in this format:

```text
FFNN 4T1002.mat 5
```

The columns are model, subject filename, and analysis value. Each process writes
two uniquely named files in its scratch directory. HTCondor remaps those files
into `analyses/subaverage/` or `analyses/data_amount/` on the access point.

Each condition produces two files named with its model, subject, condition, and
value:

- `*.summary.json`: status, parameters, timing, job metadata, aggregate metrics,
  and a traceback when the analysis failed
- `*.predictions.csv`: fold and trial-level labels, predictions, and
  probabilities

Per-subject submit files transfer only `src/`, `server_analysis/`, and that
job's subject `.mat` file. They do not transfer the repository's `.git`, local
virtual environment, GUI checkpoints, or unrelated subjects. The autoencoder
submission still transfers the complete data directory because it loads all
subjects in one process.
