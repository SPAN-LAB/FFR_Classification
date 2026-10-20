# Generic Analysis Design

Generic analysis measures how well a model generalizes to a subject that was
not used for training. It uses leave-one-subject-out (LOSO) evaluation.

## Evaluation Unit

Each run fixes one held-out subject:

- Train on other subjects; data-amount conditions may omit one training subject
  that lacks the requested number of trials per category.
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

1. Select that many raw trials per category from each eligible training subject,
   using a fixed seed so higher amounts include all trials used at lower amounts.
   Omit a training subject at values above its smallest category count.
2. Subaverage the selected trials within each subject and category (default size five).
3. Pool the processed training subjects and train a new model.
4. Evaluate the same fixed held-out test data at every amount.

The x-value means **raw training trials per category per included training subject**.
The default sweep starts at 50 and advances by 50. It stops at the largest
value for which every held-out fold retains all but at most one training subject
(at least nine of ten for an 11-subject dataset). The summary records the
included and excluded training subjects and actual raw training-trial count for
each value. Higher values may therefore use a different set of training subjects.
The top-level `training_subjects` list contains all possible training subjects;
each condition's list records those actually used.

The default subaverage size is five. Pass
`--data-amount-subaverage-size SIZE` to run the same data-amount sweep with a
different size. Explicit-size runs are kept separate under:

```text
analyses/generic/data_amount_by_subaverage/subaverage_<size>/<model>/
```

Within one run, the held-out test subject uses the selected subaverage size and
remains fixed across all data-amount values. Comparing different subaverage
sizes therefore compares both training and test examples at that size.

## Outputs

Each held-out run writes one summary JSON and one predictions CSV under:

```text
analyses/generic/<analysis>/<model>/
```
