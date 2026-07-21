"""
SPAN Lab - FFR Classification

Filename: run_eval_autoencoder.py
Description: CLI entry point for running Autoencoder LOSO evaluation on CHTC.
    Unlike other models, the Autoencoder requires all subjects to be loaded at
    once for LOSO pretraining — it cannot be parallelized per subject.

Usage:
    python -m server_analysis.run_eval_autoencoder
"""

from src.core import AnalysisPipeline


SUBJECT_FILEPATHS = [
    "4T1002.mat",
    "4T1004.mat",
    "4T1005.mat",
    "4T1006.mat",
    "4T1007.mat",
    "4T1008.mat",
    "4T1009.mat",
    "4T1010.mat",
    "4T1012.mat",
    "4T1014.mat",
    "4T1015.mat",
]

TRAINING_OPTIONS = {
    "latent_dim": 128,
    "num_epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.001,
}


def main():
    print(f"[run_eval_autoencoder] Running LOSO Autoencoder on {len(SUBJECT_FILEPATHS)} subjects")

    (
        AnalysisPipeline()
        .load_subjects(SUBJECT_FILEPATHS)
        .trim_by_timestamp(start_time=0, end_time=float("inf"))
        .subaverage(5)
        .fold(5)
        .evaluate_model(
            model_name="Autoencoder",
            training_options=TRAINING_OPTIONS,
        )
    )

    print("[run_eval_autoencoder] Done.")


if __name__ == "__main__":
    main()
