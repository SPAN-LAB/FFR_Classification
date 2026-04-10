"""
SPAN Lab - FFR Classification

Filename: run_analysis.py
Description: CLI entry point for running analyses on CHTC.
    Designed to be called once per subject so jobs can be parallelized.

Usage:
    python -m server_analysis.run_analysis \
        --model CNN \
        --subject data/4T1002.mat \
        --analysis subaverage \
        --output_dir analyses
"""

import argparse

from src.analysis.accuracy_against_subaverage_size import accuracy_against_subaverage_size
from src.analysis.accuracy_against_data_amount import accuracy_against_data_amount


TRAINING_OPTIONS = {
    "num_epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.1,
}

SUBAVERAGE_SIZES = list(range(5, 126, 5))  # 5, 10, ..., 125
DATA_AMOUNT_MIN_TRIALS = 100
DATA_AMOUNT_STRIDE = 100


def main():
    parser = argparse.ArgumentParser(description="Run FFR classification analyses for one subject.")
    parser.add_argument("--model", required=True, help="Model name, e.g. FFNN, CNN, GRU")
    parser.add_argument("--subject", required=True, help="Path to subject .mat file")
    parser.add_argument(
        "--analysis",
        required=True,
        choices=["subaverage", "data_amount"],
        help="Which analysis to run"
    )
    parser.add_argument("--output_dir", default="analyses", help="Root output directory")
    args = parser.parse_args()

    subject_filepaths = [args.subject]
    model_names = [args.model]

    if args.analysis == "subaverage":
        print(f"[run_analysis] subaverage | model={args.model} | subject={args.subject}")
        accuracy_against_subaverage_size(
            subaverage_sizes=SUBAVERAGE_SIZES,
            subject_filepaths=subject_filepaths,
            model_names=model_names,
            training_options=TRAINING_OPTIONS,
            output_folder_path=f"{args.output_dir}/subaverage",
            include_null_case=True,
            defer_subject_loading=True,
        )

    elif args.analysis == "data_amount":
        print(f"[run_analysis] data_amount | model={args.model} | subject={args.subject}")
        accuracy_against_data_amount(
            min_trials=DATA_AMOUNT_MIN_TRIALS,
            stride=DATA_AMOUNT_STRIDE,
            subject_filepaths=subject_filepaths,
            model_names=model_names,
            training_options=TRAINING_OPTIONS,
            output_folder_path=f"{args.output_dir}/data_amount",
            defer_subject_loading=True,
        )

    print("[run_analysis] Done.")


if __name__ == "__main__":
    main()
