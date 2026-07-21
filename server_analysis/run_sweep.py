"""
SPAN Lab - FFR Classification

Filename: run_sweep.py
Description: Grid search over learning_rate and weight_decay for a given model and subject.
    Run once per model to find the best hyperparameters. Results print to stdout (.out log on CHTC).

Usage:
    python -m server_analysis.run_sweep \
        --model CNN \
        --subject 4T1002.mat
"""

import argparse
from src.core import AnalysisPipeline, EEGSubject
from src.models.utils import find_model


BASE_OPTIONS = {
    "num_epochs": 50,
    "batch_size": 64,
    "patience": 10,
    "min_delta": 0.001,
    "validation_ratio": 0.2,
}

LEARNING_RATES = [1e-3, 5e-4, 1e-4, 5e-5]
WEIGHT_DECAYS  = [1e-2, 1e-3, 1e-4]


def main():
    parser = argparse.ArgumentParser(description="Grid search over learning_rate and weight_decay.")
    parser.add_argument("--model", required=True, help="Model name, e.g. CNN, FFNN, GRU")
    parser.add_argument("--subject", required=True, help="Path to subject .mat file")
    args = parser.parse_args()

    print(f"[run_sweep] model={args.model} | subject={args.subject}")
    print(f"[run_sweep] Grid: lr={LEARNING_RATES} | wd={WEIGHT_DECAYS}")
    print("-" * 60)

    # Load subject once, reuse across all combos
    pipeline = (
        AnalysisPipeline()
        .load_subjects(args.subject)
        .trim_by_timestamp(start_time=0, end_time=float("inf"))
        .subaverage(5)
        .fold(5)
    )
    subject = pipeline.subjects[0]

    concrete_model = find_model(args.model)

    best_val_loss = float("inf")
    best_lr = None
    best_wd = None

    for lr in LEARNING_RATES:
        for wd in WEIGHT_DECAYS:
            training_options = {**BASE_OPTIONS, "learning_rate": lr, "weight_decay": wd}

            model = concrete_model(training_options)
            model.set_subject(subject)

            # Use the first fold's training set to get a val loss estimate
            train_trials = []
            for fold_i, fold in enumerate(subject.folds):
                if fold_i != 0:
                    train_trials += fold

            model._core_train(
                trials=train_trials,
                validation_trials=BASE_OPTIONS["validation_ratio"],
                num_epochs=BASE_OPTIONS["num_epochs"],
                batch_size=BASE_OPTIONS["batch_size"],
                learning_rate=lr,
                weight_decay=wd,
                min_delta=BASE_OPTIONS["min_delta"],
                patience=BASE_OPTIONS["patience"],
            )

            val_loss = model._lowest_loss
            print(f"lr={lr:.0e} | wd={wd:.0e} | val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lr = lr
                best_wd = wd

    print("-" * 60)
    print(f"[run_sweep] Best: lr={best_lr:.0e} | wd={best_wd:.0e} | val_loss={best_val_loss:.4f}")
    print("[run_sweep] Done.")


if __name__ == "__main__":
    main()
