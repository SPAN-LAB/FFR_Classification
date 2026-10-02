# Generic Analysis Design

Generic analysis measures how well a model generalizes to a subject that was
not used for training. It uses leave-one-subject-out (LOSO) evaluation.

## Evaluation Unit

Each run fixes one held-out subject:

- Train on every other subject.
- Test only on the held-out subject.
- Repeat until every subject has been held out once, for 11 folds total.
- Average held-out accuracies when comparing models.

Neural models train for exactly 50 epochs with no validation split or early
stopping. LDA and SVM use their normal non-epoch-based training.

## Shared Preprocessing

- Trim every waveform to 50-250 ms.
- Derive label mappings and model dimensions from training subjects.
- Only average trials together that are from the same subject.
- Save predictions only for the held-out subject.

## Subaverage Analysis

For each configured subaverage size:

1. Subaverage each subject and category independently.
2. Pool the processed trials from all training subjects.
3. Train a new model on the pooled training trials.
4. Evaluate the separately processed held-out subject.

## Data-Amount Analysis

For each configured amount:

1. Select that many raw trials from each training subject, balanced by label.
2. Subaverage the selected training trials by five.
3. Pool the processed training subjects and train a new model.
4. Evaluate the same fixed held-out test data at every amount.

The x-value means **raw training trials per training subject**, not total pooled
trials.

## Outputs

Each held-out run writes one summary JSON and one predictions CSV under:

```text
analyses/generic/<analysis>/<model>/
```
