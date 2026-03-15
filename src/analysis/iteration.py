import pickle 
from .utils import strip_data_away
from ..constants.defaults import NUM_FOLDS, TRIM_START_TIME, TRIM_END_TIME
from ..time import TimeKeeper
from ..printing.logging import log, is_empty
from datetime import datetime, timezone

def iteration(
    model_name, 
    training_options,
    subaverage_size, 
    pipeline_copy, 
    write_directory,
    pkl_filename_prefix,
    iteration_quantifier,
    independent_var_name: str,
    time_keeper: TimeKeeper,
    time_log_filename: str
):
    
    p = (
        pipeline_copy
        .trim_by_timestamp(start_time=TRIM_START_TIME, end_time=TRIM_END_TIME)
        .subaverage(size=subaverage_size)
        .fold(num_folds=NUM_FOLDS)
    )
    
    # NOTE: We are measuring strictly the time spent to evaluate the model,
    # hence the splitting of the chain of AnalysisPipeline methods.
    time_keeper.reset()
    time_keeper.start()
    
    p = (
        p
        .evaluate_model(
            model_name=model_name,
            training_options=training_options
        )
    )
    
    time_keeper.stop()
    
    subject = p.subjects[0]
    # So that storage size is smaller, since we only need predictions
    strip_data_away(subject)
    
    # Create needed folders and save
    full = write_directory / f"{pkl_filename_prefix}{iteration_quantifier}.pkl"
    full.parent.mkdir(parents=True, exist_ok=True)
    with full.open("wb") as file:
        pickle.dump(subject, file)
    
    # Create the header
    time_log_path = write_directory / time_log_filename
    if not time_log_path.exists() or is_empty(time_log_path):
        log(
            f"{independent_var_name},time(seconds),completion-time(utc)",
            time_log_path
        )
    
    # Create a guide to the time log
    pitl_description_filename = "times-log-note.txt"
    pitl_description_filepath = write_directory / pitl_description_filename
    if not pitl_description_filepath.exists() or is_empty(pitl_description_filepath):
        pitl_description = (
            "Each value in the \"duration\" column in times-log.csv indicates "
            "the time elapsed during the evaluation of the model on this "
            "subject for each iteration of the investigation parameter."
        )
        log(pitl_description.strip(), pitl_description_filepath)
    
    
    log(
        (
            f"{iteration_quantifier}"
            + f",{time_keeper.accumulated_duration}"
            + f",{datetime.now(timezone.utc).isoformat()}"
        ),
        time_log_path
    )
    time_keeper.accumulated_duration