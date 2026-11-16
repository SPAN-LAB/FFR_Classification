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
    FFNN/
      FFNN.4T1002.mat.subaverage.summary.json
      FFNN.4T1002.mat.subaverage.predictions.csv
  data_amount/
    FFNN/
      FFNN.4T1002.mat.data_amount.summary.json
      FFNN.4T1002.mat.data_amount.predictions.csv
```

The summary contains one condition record per value, including accuracy,
duration, parameters, status, and traceback. The CSV contains fold-level trial
predictions for every successful value.

For `data_amount`, values start at 100 and increase by 100 raw training trials
per cross-validation fold. The five raw test folds remain fixed across the
sweep. Training and test trials are subaveraged separately before evaluation.

## Data amount at other subaverage sizes

`--data-amount-subaverage-size SIZE` runs the same data-amount sweep with the
training sample and test folds subaveraged at `SIZE` instead of 5:

```bash
python -m server_analysis.run_analysis \
  --model FFNN \
  --subject /path/to/4T1002.mat \
  --analysis data_amount \
  --data-amount-subaverage-size 10
```

Results are written separately from the default sweep:

```text
analyses/data_amount_by_subaverage/subaverage_<size>/<model>/<model>.<subject>.data_amount.sa<size>.summary.json
```

The x-values, folds, and sampled raw trials are identical at every size; only
the averaging changes. Plot each size separately. Large sizes leave few
training examples at the smallest amounts: 100 raw trials is 25 per category,
so sizes 15 to 25 give one example per category. LDA cannot fit that, and SVM
needs at least three per category, so those conditions are recorded as
failures while the rest of the sweep continues.
