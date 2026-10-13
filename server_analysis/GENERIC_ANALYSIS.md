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

1. Select that many raw trials per category from each training subject, using a
   fixed seed so higher amounts include all trials used at lower amounts.
2. Subaverage the selected trials within each subject and category (default size five).
3. Pool the processed training subjects and train a new model.
4. Evaluate the same fixed held-out test data at every amount.

The x-value means **raw training trials per category per training subject**.
The sweep stops at the smallest available category count across all subjects.
For example, with four categories and ten training subjects, an x-value of 100
selects 4,000 raw training trials before subaveraging.

The default subaverage size is five. Pass
`--data-amount-subaverage-size SIZE` to run the same data-amount sweep with a
different size. Explicit-size runs are kept separate under:

```text
analyses/generic/data_amount_by_subaverage/subaverage_<size>/<model>/
```

Within one run, the held-out test subject uses the selected subaverage size and
remains fixed across all data-amount values. Comparing different subaverage
sizes therefore compares both training and test examples at that size.

### Data Amount Across Subaverage Sizes on CHTC

All generic data-amount results need to be rerun under this value definition.
Pass all six sizes explicitly so an older copy of the gitignored CHTC submit
script also includes size 5.

```bash
./submit_generic_data_amount_subaverages.sh --subjects-file subjects.txt --subaverage-sizes 1,5,10,15,20,25 all
```

SVM uses CPU jobs and must be submitted separately:

```bash
./submit_generic_data_amount_subaverages.sh --subjects-file subjects.txt --subaverage-sizes 1,5,10,15,20,25 --submit-file generic_data_amount_subaverages_svm.sub SVM
```

Use `--dry-run` to inspect the generated manifest without submitting jobs.

## Outputs

Each held-out run writes one summary JSON and one predictions CSV under:

```text
analyses/generic/<analysis>/<model>/
```
