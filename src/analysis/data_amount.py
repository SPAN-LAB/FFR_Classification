from src.analysis import accuracy_against_data_amount
from src.analysis.utils import strip_data_away
from src.core import AnalysisPipeline, EEGSubject, EEGTrial
from src.constants import defaults
from src.core.utils.sampling import sds2
from src.models.utils.resolver import find_model
from src.time import TimeKeeper
from src.printing.logging import log, is_empty
from math import ceil
from copy import deepcopy
from pathlib import Path
import pickle


def analyze(model_name: str, subject_filepath: str, output_dirpath: str):

    MODEL_NAMES = [model_name]
    SUBJECT_FILEPATHS = [subject_filepath]
    OUTPUT_DIR_PATH = output_dirpath
    MIN_TRIALS = 100
    STRIDE = 100 # Trial number is incremented by this amount
    DEFER_SUBJECT_LOADING = False
    TRAINING_OPTIONS = {
        "num_epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.1
    }

    accuracy_against_data_amount(
        min_trials=MIN_TRIALS,
        stride=STRIDE,
        subject_filepaths=SUBJECT_FILEPATHS,
        model_names=MODEL_NAMES,
        training_options=TRAINING_OPTIONS,
        output_folder_path=OUTPUT_DIR_PATH,
        defer_subject_loading=DEFER_SUBJECT_LOADING
    )

def generic_analyze(
    model_name: str, 
    subject_filepath: str, 
    all_subject_filepaths: list[str],
    output_dirpath: str,
):
    
    MIN_TRIALS = 100
    STRIDE = 100
    
    # Set up output directory paths
    independent_var_name = "data_amount"
    pkl_filename_prefix = f"{independent_var_name}"
    subject_filename = Path(subject_filepath).stem
    write_directory = (
        Path(output_dirpath)
        / independent_var_name
        / f"generic_{model_name}"
        / f"withholding_{subject_filename}"
    )
    
    # Read in subject data from files
    train_subject_filepaths = [fp for fp in all_subject_filepaths 
        if Path(fp).absolute() != Path(subject_filepath).absolute()]
    base_train_pipeline = AnalysisPipeline()
    base_train_pipeline.load_subjects(train_subject_filepaths)
    base_train_pipeline.trim_by_timestamp(start_time=defaults.TRIM_START_TIME, end_time=defaults.TRIM_END_TIME)
    base_test_pipeline = AnalysisPipeline()
    base_test_pipeline.load_subjects(subject_filepath)
    base_test_pipeline.trim_by_timestamp(start_time=defaults.TRIM_START_TIME, end_time=defaults.TRIM_END_TIME)
    
    # Find the max data amount
    min_trials_per_tone = EEGSubject.min_trials_per_label(
        subjects=[*base_train_pipeline.subjects, *base_test_pipeline.subjects]
    )
    max_data_amount = min(list(min_trials_per_tone.values())) * 4
    print(f"{max_data_amount = }")
    
    for data_amount in range(MIN_TRIALS, max_data_amount + 1, STRIDE):
        
        # Note the 3 versions of `test_pipeline`
        # `pre_test_pipeline` is used to find the accuracy on the withheld 
        #     subject right after training 
        # `post_test_pipeline` is used to find the cross validation accuracy
        #     of the model on the withheld subject. Each fold, the model is 
        #     initialized with the weights found from the pre-training step. 
        # `test_pipeline` is used to find the cross validation accuracy on the 
        #     withheld subject, using a model that hasn't been pre-trained
        train_pipeline     = deepcopy(base_train_pipeline)
        pre_test_pipeline  = deepcopy(base_test_pipeline)
        post_test_pipeline = deepcopy(base_test_pipeline)
        test_pipeline      = deepcopy(base_test_pipeline)
    
        pre_test_pipeline.subaverage(defaults.SUBAVERAGE_SIZE)
        pre_test_pipeline.fold(defaults.NUM_FOLDS)
        post_test_pipeline.subaverage(defaults.SUBAVERAGE_SIZE)
        post_test_pipeline.fold(defaults.NUM_FOLDS)
        test_pipeline.subaverage(defaults.SUBAVERAGE_SIZE)
        test_pipeline.fold(defaults.NUM_FOLDS)
        
        # Gather train, validation, and test sets
        train_trials = []
        validation_trials = []
        test_trials = []
        for subject in train_pipeline.subjects:
            validation_ratio = defaults.VALIDATION_RATIO
            subject.trials = sds2(subject.trials, num_trials=data_amount)
            # print(f"Befoore subaveraging we have {len(subject.trials)} trials")
            subject.subaverage(defaults.SUBAVERAGE_SIZE)
            # print(f"After subaveraging we have {len(subject.trials)} trials")
            validation_trials += sds2(
                trials=subject.trials, 
                num_trials=ceil(len(subject.trials) * validation_ratio)
            )
            train_trials += [
                trial for trial in subject.trials if trial not in validation_trials
            ]
        test_trials = pre_test_pipeline.subjects[0].trials
        
        # print(f"{len(train_trials) = }")
        # print(f"{len(validation_trials) = }")
        # print(f"{len(test_trials) = }")
        
        # ML stuff
        tk_pre = TimeKeeper(); tk_pre.start()
        model = find_model(model_name)(training_options=defaults.TRAINING_OPTIONS)
        model.set_subject(subject=pre_test_pipeline.subjects[0]) # NOTE: dummy
        model.train(
            trials=train_trials,
            validation_trials=validation_trials
        )
        model.infer(trials=test_trials)
        tk_pre.stop()
        
        # Writing the outputs to the directory
        
        # Data
        iteration_quantifier = data_amount
        full = write_directory / f"pre_{pkl_filename_prefix}_{iteration_quantifier}.pkl"
        full.parent.mkdir(parents=True, exist_ok=True)
        with full.open("wb") as file:
            strip_data_away(pre_test_pipeline.subjects[0])
            pickle.dump(pre_test_pipeline.subjects[0], file)
            
        print(f"{EEGTrial.get_accuracy(trials=test_trials) = }")
        print(f"{EEGTrial.get_per_label_accuracy(trials=test_trials) = }")
        
        
        # Cross validation on the withheld subject using the pre-trained model
        tk_post = TimeKeeper(); tk_post.start()
        best_weights = deepcopy(model._get_best())
        pre_trained_model = find_model(model_name)(training_options=defaults.TRAINING_OPTIONS)
        pre_trained_model.set_subject(subject=post_test_pipeline.subjects[0])
        acc = pre_trained_model.evaluate(
            folded_trials=post_test_pipeline.subjects[0].folds,
            base_state=best_weights
        )
        print(f"Accuracy on pre-trained model: {acc}")
        tk_post.stop()
        
        # Writing the outputs to the directory
        iteration_quantifier = data_amount
        full = write_directory / f"post_{pkl_filename_prefix}_{iteration_quantifier}.pkl"
        full.parent.mkdir(parents=True, exist_ok=True)
        with full.open("wb") as file:
            strip_data_away(post_test_pipeline.subjects[0])
            pickle.dump(post_test_pipeline.subjects[0], file)
        
        # Cross validation on the withheld subjects without the pre-trained model
        tk_control = TimeKeeper(); tk_control.start()
        blank_model = find_model(model_name)(training_options=defaults.TRAINING_OPTIONS)
        blank_model.set_subject(subject=test_pipeline.subjects[0])
        acc = blank_model.evaluate(
            folded_trials=test_pipeline.subjects[0].folds
        )
        print(f"Accuracy without pre-trained model: {acc}")
        tk_control.stop()
        
        # Writing the outputs to the directory
        iteration_quantifier = data_amount
        full = write_directory / f"control_{pkl_filename_prefix}_{iteration_quantifier}.pkl"
        full.parent.mkdir(parents=True, exist_ok=True)
        with full.open("wb") as file:
            strip_data_away(test_pipeline.subjects[0])
            pickle.dump(test_pipeline.subjects[0], file)
        
        # Times
        log_filename = "times_log.csv"
        log_filepath = write_directory / log_filename
        if not log_filepath.exists() or is_empty(log_filepath):
            log("iteration_quantifier,pre-training(s),cv_on_withheld_using_pretrained(s),cv_on_withheld_without_pretrained(s)", log_filepath)
        log(f"{iteration_quantifier},{tk_pre.accumulated_duration},{tk_post.accumulated_duration},{tk_control.accumulated_duration}", log_filepath)
        