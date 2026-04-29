from enum import StrEnum
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
        

class AnalysisType(StrEnum):
    DATA_AMOUNT = "data_amount"
    SUBAVERAGE_SIZE = "subaverage_size"

def analyze2(*,
    analysis_type: AnalysisType | str,
    is_generic: str,
    model_name: str,
    subaverage_sizes: list[int],
    data_amounts: list[int],
    subject_filepath: str,
    all_subject_filepaths: list[str] | None,
    output_root_dirpath: str
):
    """
    analysis_type         "data_amount" or "subaverage_size"
    is_generic            Whether to create and test generic models (requires `all_subject_filepaths` to be not None)
    subaverage_sizes      The subaverage sizes to use
    data_amounts          The data amounts to use. If empty, no sampling is performed
    subject_filepath      The path to the file containing main subject data
    all_subject_filepaths The paths to all the subjects' data. Ignored if not generic
    output_root_dirpath   Where outputs are written to
    """

    # Validate input
    if analysis_type == AnalysisType.DATA_AMOUNT and len(data_amounts) == 0:
        raise ValueError("data_amount parameter shouldn't be empty")
    if len(data_amounts) == 0:
        data_amounts = [-1]
    if analysis_type == AnalysisType.SUBAVERAGE_SIZE and len(subaverage_sizes) == 1:
        print("Warning: Only 1 subaverage size was provided")
    if is_generic and (all_subject_filepaths is None or len(all_subject_filepaths) == 1):
        raise ValueError("all_subject_filepaths parameter is invalid")
    if is_generic and subject_filepath not in all_subject_filepaths:
        raise ValueError("subject_filepath not found in all_subject_filepaths")

    # Create the directory where outputs are written to
    base_dirpath = Path(output_root_dirpath)
    base_dirpath.mkdir(parents=True, exist_ok=True)
    full_dirpath = (
        base_dirpath
        / analysis_type
        / ("generic" if is_generic else "specific")
        / model_name
        / Path(subject_filepath).stem
    )

    # Load subjects
    base_subject_pipelines: dict[str, AnalysisPipeline] = {}
    if not is_generic:
        all_subject_filepaths = [subject_filepath]
    for filepath in all_subject_filepaths:
        pipeline = AnalysisPipeline().load_subjects(filepath)
        pipeline.trim_by_timestamp(start_time=defaults.TRIM_START_TIME, end_time=defaults.TRIM_END_TIME)
        base_subject_pipelines[filepath] = pipeline

    for data_amount in data_amounts:
        for subaverage_size in subaverage_sizes:

            if analysis_type == AnalysisType.DATA_AMOUNT:
                iteration_quantifier = data_amount
            else:
                iteration_quantifier = subaverage_size

            subject_pipelines = deepcopy(base_subject_pipelines)

            subject_pipeline_of_focus = subject_pipelines[subject_filepath]

            # Filter `data_amount` number of trials from the appropriate subjects
            if data_amount != -1:
                print("data amounting")
                if is_generic:
                    for filepath, pipeline in subject_pipelines.items():
                        if filepath != subject_filepath:
                            pipeline.subjects[0].trials = sds2(
                                pipeline.subjects[0].trials,
                                data_amount
                            )
                else: # specific model
                    subject_pipeline_of_focus.subjects[0].trials = sds2(
                        subject_pipeline_of_focus.subjects[0].trials,
                        data_amount
                    )
            
            for pipeline in subject_pipelines.values():
                pipeline.subaverage(subaverage_size)

            if is_generic:
                train_trials = []
                validation_trials = []
                subject_of_focus_on_generic_model = deepcopy(subject_pipeline_of_focus.subjects[0])
                subject_of_focus_using_pre_trained = deepcopy(subject_pipeline_of_focus.subjects[0])
                subject_of_focus_using_pre_trained.fold(defaults.NUM_FOLDS)
                
                for filepath, pipeline in subject_pipelines.items():
                    if filepath == subject_filepath:
                        continue
                    num_validation_trials = ceil(len(pipeline.subjects[0].trials) * defaults.VALIDATION_RATIO)
                    subject_validation_trials = sds2(
                        pipeline.subjects[0].trials,
                        num_validation_trials
                    )
                    validation_trials += subject_validation_trials
                    train_trials += [
                        trial for trial in pipeline.subjects[0].trials if trial not in subject_validation_trials
                    ]

                # Training

                # Initial training on all but focused subject
                generic_model = find_model(model_name)(training_options=defaults.TRAINING_OPTIONS)
                generic_model.set_subject(subject_of_focus_on_generic_model)
                generic_model.train(
                    trials=train_trials,
                    validation_trials=validation_trials
                )

                # Testing on the withheld subject
                acc = generic_model.infer() # Infer on subject_of_focus_on_generic_model
                print(f"Accuracy: {(acc * 100):.1f}%")

                # Save the predictions of the not-pretrain subject
                not_pretrain_save_dirpath = full_dirpath / "not_pretrain"
                not_pretrain_save_dirpath.mkdir(parents=True, exist_ok=True)
                not_pretrain_save_filepath = not_pretrain_save_dirpath / f"{iteration_quantifier}.pkl"
                not_pretrain_csv_filepath = not_pretrain_save_dirpath / f"{iteration_quantifier}.csv"
                subject_of_focus_on_generic_model.save_predictions_to_csv(not_pretrain_csv_filepath)
                with open(not_pretrain_save_filepath, "wb") as file:
                    strip_data_away(subject_of_focus_on_generic_model)
                    pickle.dump(subject_of_focus_on_generic_model, file)

                # Cross validation on the withheld subject
                weights = generic_model._get_best()
                fine_tuned_model = find_model(model_name)(training_options=defaults.TRAINING_OPTIONS)
                fine_tuned_model.set_subject(subject_of_focus_using_pre_trained)
                acc = fine_tuned_model.evaluate(base_state=weights)
                print(f"Accuracy: {(acc * 100):.1f}%")

                # Save the predictions of the use-pretrain subject
                use_pretrain_save_dirpath = full_dirpath / "use_pretrain"
                use_pretrain_save_dirpath.mkdir(parents=True, exist_ok=True)
                use_pretrain_save_filepath = use_pretrain_save_dirpath / f"{iteration_quantifier}.pkl"
                use_pretrain_csv_filepath = use_pretrain_save_dirpath / f"{iteration_quantifier}.csv"
                subject_of_focus_using_pre_trained.save_predictions_to_csv(use_pretrain_csv_filepath)
                with open(use_pretrain_save_filepath, "wb") as file:
                    strip_data_away(subject_of_focus_using_pre_trained)
                    pickle.dump(subject_of_focus_using_pre_trained, file)

            specific_model = find_model(model_name)(training_options=defaults.TRAINING_OPTIONS)
            specific_model.set_subject(subject_pipeline_of_focus.subjects[0])
            subject_pipeline_of_focus.subjects[0].fold(defaults.NUM_FOLDS)
            acc = specific_model.evaluate()
            print(f"Accuracy: {(acc * 100):.1f}%")

            # Save the predictions of the use-pretrain subject
            if is_generic:
                control_save_dirpath = full_dirpath / "control"
            else:
                control_save_dirpath = full_dirpath
            control_save_dirpath.mkdir(parents=True, exist_ok=True)
            control_save_filepath = control_save_dirpath / f"{iteration_quantifier}.pkl"
            control_csv_filepath = control_save_dirpath / f"{iteration_quantifier}.csv"
            subject_pipeline_of_focus.subjects[0].save_predictions_to_csv(control_csv_filepath)
            with open(control_save_filepath, "wb") as file:
                strip_data_away(subject_pipeline_of_focus.subjects[0])
                pickle.dump(subject_pipeline_of_focus.subjects[0], file)
