import pickle
from pathlib import Path

from .utils import strip_data_away
from ..constants.defaults import NUM_FOLDS, TRIM_START_TIME, TRIM_END_TIME
from ..time import TimeKeeper
from ..printing.logging import log, is_empty

def iteration(
    model_name,
    training_options,
    subaverage_size,
    pipeline_copy,
    write_directory,
    pkl_filename_prefix,
    iteration_quantifier
):

    tk = TimeKeeper(); tk.reset(); tk.start();
    p = (
        pipeline_copy
        .trim_by_timestamp(start_time=TRIM_START_TIME, end_time=TRIM_END_TIME)
        .subaverage(size=subaverage_size)
        .fold(num_folds=NUM_FOLDS)
        .evaluate_model(
            model_name=model_name,
            training_options=training_options
        )
    )
    tk.stop()
    _save_iteration(p, write_directory, pkl_filename_prefix, iteration_quantifier, tk)


def training_amount_iteration(
    model_name,
    training_options,
    training_amount,
    pipeline_copy,
    write_directory,
    pkl_filename_prefix,
):
    """Evaluate one training amount against folds created before the sweep."""
    tk = TimeKeeper(); tk.reset(); tk.start()
    p = pipeline_copy.evaluate_model_with_training_amount(
        model_name=model_name,
        training_options=training_options,
        training_amount=training_amount,
    )
    tk.stop()
    _save_iteration(p, write_directory, pkl_filename_prefix, training_amount, tk)


def _save_iteration(
    pipeline,
    write_directory,
    pkl_filename_prefix,
    iteration_quantifier,
    time_keeper,
):
    subject = pipeline.subjects[0]

    # So that storage size is smaller, since we only need predictions
    strip_data_away(subject)

    # Create needed folders and save
    full = write_directory / f"{pkl_filename_prefix}_{iteration_quantifier}.pkl"
    full.parent.mkdir(parents=True, exist_ok=True)
    with full.open("wb") as file:
        pickle.dump(subject, file)

    # Save times
    log_filename = "times_log.csv"
    log_filepath = write_directory / log_filename
    if not log_filepath.exists() or is_empty(log_filepath):
        log("iteration_quantifier,duration(s)", log_filepath)
    log(f"{iteration_quantifier},{time_keeper.accumulated_duration}", log_filepath)
