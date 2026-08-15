# Running Analyses

There is one canonical analysis command for both local execution and CHTC:

```bash
python -m server_analysis.run_analysis \
  --model FFNN \
  --subject /path/to/4T1002.mat \
  --analysis subaverage
```

Omitting `--value` runs the complete configured sweep. Supplying `--value`
runs only that condition and may be repeated:

```bash
python -m server_analysis.run_analysis \
  --model FFNN \
  --subject /path/to/4T1002.mat \
  --analysis subaverage \
  --value 1 --value 5 --value 10
```

All shared settings live in `src/analysis/settings.py`, including trimming,
fold count, default subaverage values, data-amount stride, and model training
options.

## CHTC

On the CHTC access point, update `subjects.txt`, then run from
`jobs_config/ffr_test_run`:

```bash
./submit_analysis.sh FFNN
```

That submits every subject for both `subaverage` and `data_amount`. Select only
one type with:

```bash
./submit_analysis.sh FFNN --analyses subaverage
```

Pass multiple model names, or every model with explicit shared training
settings:

```bash
./submit_analysis.sh FFNN CNN PitchCNN
./submit_analysis.sh all
```

Use `--dry-run` to inspect `analysis_jobs.txt` without submitting:

```bash
./submit_analysis.sh FFNN --dry-run
```

Each Condor process handles one model, subject, and analysis type. The runner
determines all valid values internally. Results return to:

```text
analyses/
  subaverage/
    FFNN.4T1002.mat.subaverage.summary.json
    FFNN.4T1002.mat.subaverage.predictions.csv
  data_amount/
    FFNN.4T1002.mat.data_amount.summary.json
    FFNN.4T1002.mat.data_amount.predictions.csv
```

The summary contains one condition record per value, including accuracy,
duration, parameters, status, and traceback. The CSV contains fold-level trial
predictions for every successful value.
