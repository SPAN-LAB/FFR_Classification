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
from src.core import AnalysisPipeline


TRAINING_OPTIONS = {
    "num_epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.1,
}


def main():
    parser = argparse.ArgumentParser(description="Run FFR model eval for one subject.")
    parser.add_argument("--model", required=True, help="Model name, e.g. FFNN, CNN, GRU")
    parser.add_argument("--subject", required=True, help="Path to subject .mat file")
    args = parser.parse_args()

    print(f"[run_eval] model={args.model} | subject={args.subject}")

    (
        AnalysisPipeline()
        .load_subjects(args.subject)
        .trim_by_timestamp(start_time=0, end_time=float("inf"))
        .subaverage(5)
        .fold(5)
        .evaluate_model(
            model_name=args.model,
            training_options=TRAINING_OPTIONS,
        )
    )

    print("[run_eval] Done.")


if __name__ == "__main__":
    main()
