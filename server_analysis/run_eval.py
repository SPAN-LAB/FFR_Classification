"""
SPAN Lab - FFR Classification

Filename: run_eval.py
Description: CLI entry point for running model training and evaluation on CHTC.
    Designed to be called once per subject so jobs can be parallelized.

Usage:
    python -m server_analysis.run_eval \
        --model CNN \
        --subject 4T1002.mat
"""

import argparse
from pathlib import Path

from src.core import AnalysisPipeline


TRAINING_OPTIONS = {
    "CNN": {
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
    },
    "FFNN": {
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
    },
    "PitchCNN": {
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "patience": 50,
    },
    "EEGNeXFFR": {
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "patience": 50,
        "sinc_frontend": False,
    },
}

DEFAULT_TRAINING_OPTIONS = {
    "num_epochs": 50,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "patience": 50,
}


def main():
    parser = argparse.ArgumentParser(description="Run FFR model eval for one subject.")
    parser.add_argument(
        "--model", required=True, help="Model name, e.g. FFNN, CNN, GRU"
    )
    parser.add_argument("--subject", required=True, help="Path to subject .mat file")
    parser.add_argument(
        "--generic",
        action="store_true",
        help="Train a subject-independent model (LOSO): hold out --subject, "
             "train on every other subject in --data-dir, test on --subject.",
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory of .mat files to pool for --generic training. Default: cwd.",
    )
    args = parser.parse_args()

    opts = TRAINING_OPTIONS.get(args.model, DEFAULT_TRAINING_OPTIONS)

    if args.generic:
        held_out = Path(args.subject).stem
        print(f"[run_eval] generic LOSO | model={args.model} | held out={held_out} | data_dir={args.data_dir}")
        (
            AnalysisPipeline()
            .load_subjects(args.data_dir)
            .trim_by_timestamp(start_time=0, end_time=float("inf"))
            .subaverage(1)
            .evaluate_generic_model(
                model_name=args.model,
                training_options=opts,
                only_held_out=[held_out],
            )
        )
    else:
        print(f"[run_eval] per-subject | model={args.model} | subject={args.subject}")
        (
            AnalysisPipeline()
            .load_subjects(args.subject)
            .trim_by_timestamp(start_time=0, end_time=float("inf"))
            .subaverage(1)
            .fold(5)
            .evaluate_model(model_name=args.model, training_options=opts)
        )

    print("[run_eval] Done.")


if __name__ == "__main__":
    main()
