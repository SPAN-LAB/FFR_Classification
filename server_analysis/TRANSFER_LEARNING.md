# Transfer Learning Analysis Design

Transfer analysis pre-trains a model on every other subject, then fine-tunes it
on a small number of the held-out subject's own trials (calibration trials). It
measures how accuracy on a new subject changes with the amount of that
subject's data when training starts from the pre-trained model. It is separate
from per-subject cross-validation and from generic leave-one-subject-out
analysis (`GENERIC_ANALYSIS.md`).

Supported models are EEGNet, EEGNeX, EEGNeXFFR, FFNN, RNN, LSTM, and GRU. LDA
and SVM cannot be fine-tuned.

## Evaluation Unit

Each run fixes one model and one held-out subject, and covers every calibration
value. Repeat for every subject and average the held-out accuracies at each
value when comparing models.

## Protocol

For one held-out subject:

1. Trim every waveform to 50-250 ms.
2. Build the label map from the pre-training subjects only.
3. Subaverage each pre-training subject at size five, pool them, and train a new
   model for 50 epochs with no validation split. This happens once per run.
4. Fold the held-out subject's raw trials into five fixed folds.
5. For each calibration value N and each fold:
   1. Sample N raw trials, balanced across labels, from the other four folds.
      The seed matches the per-subject data-amount analysis, so equal N selects
      the same trials, and higher N includes every trial used at lower N.
   2. Subaverage the calibration sample and the test fold separately at size five.
   3. Load the pre-trained weights into a new model and fine-tune every layer
      for 50 epochs at the pre-training learning rate, with no validation
      split or schedule.
   4. Predict the test fold.
6. Accuracy for each N is pooled over the five test folds.

Every fold and every N starts from the same pre-trained weights; fine-tuning
never carries over between folds or values.

## Calibration Values

The x-value means **raw calibration trials per cross-validation fold from the
held-out subject before subaveraging**, balanced across labels. The default
values are 20, 40, 60, 80, 100, 200, 300, 400, 500, 800, 1000, 1500, 2000, and
3000. They are multiples of 20 (four labels times subaverage size five), so
subaveraging drops no trials. A run fails before pre-training if a requested
value exceeds the held-out subject's balanced calibration pool.

## Pre-trained Baseline

**N = 0 is the pre-trained baseline for this protocol**: the pre-trained model,
without fine-tuning, evaluated on the same five test folds. It is not identical
to the generic subaverage result for the same subject, because generic analysis
subaverages the held-out subject as a whole while this protocol subaverages each
test fold separately. Compare N = 0 only with other values from this analysis.

## Pre-trained Checkpoints

Each run saves its pre-trained weights and their metadata, written before any
fine-tuning starts. Passing `--pretrained-weights` to a later run skips
pre-training and reuses them. A checkpoint is only loaded if it was trained for
the same model, held-out subject, pre-training subjects and trial counts, label
map, input length, training options, subaverage size, trim window, and data
variable; otherwise the run fails rather than pre-training again. When
pre-training fails, the checkpoint metadata records the error and the weights
file is empty.

## Outputs

Each run writes one summary JSON, one predictions CSV, and the checkpoint files
`<prefix>.pretrained.pt` (weights) and `<prefix>.pretrained.json` (status and
metadata) under:

```text
analyses/generic/transfer/<model>/
```

Runs of selected values only (`--value`) are written under
`analyses/generic/transfer_partial/<model>/`, with the values in the filename
(for example `.generic.transfer.v20-100`), so they never overwrite full runs or
partial runs of other values.
