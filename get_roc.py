from src.core import AnalysisPipeline, PipelineState
from src.core import EEGSubject
from src.core.plots import plot_roc_curve, plot_confusion_matrix

def save_subject_csv(subject: EEGSubject, filepath):
    with open(filepath) as file:
        file.write("Subject, Trial_Index, Raw_Label, Mapped_Label, Prediction, Pred_Distribution\n")
        for trial in EEGSubject.trials:
            file.write(f"{EEGSubject.name}, {trial.trial_index}, {trial.raw_label}, {trial.mapped_label}, {trial.prediction}, {trial.prediction_distribution.__str__()}\n")
    return

PATHS = [
    "/mnt/z/projects/trial_classification/4tone_cell/4T1002.mat",
    "/mnt/z/projects/trial_classification/4tone_cell/4T1004.mat",
    "/mnt/z/projects/trial_classification/4tone_cell/4T1005.mat",
    "/mnt/z/projects/trial_classification/4tone_cell/4T1006.mat",
    "/mnt/z/projects/trial_classification/4tone_cell/4T1007.mat",
    "/mnt/z/projects/trial_classification/4tone_cell/4T1008.mat",
    "/mnt/z/projects/trial_classification/4tone_cell/4T1009.mat",
    "/mnt/z/projects/trial_classification/4tone_cell/4T1010.mat",
    "/mnt/z/projects/trial_classification/4tone_cell/4T1012.mat",
    "/mnt/z/projects/trial_classification/4tone_cell/4T1014.mat",
    "/mnt/z/projects/trial_classification/4tone_cell/4T1015.mat"
]
MODELS = ["FFNN"]

for model in MODELS:
    for path in PATHS:
        print(f"Training {model} for {path.split("/")[-1]}\n")

        subaverage_and_fold_result = PipelineState()
        trained_data = PipelineState()

        p = (
            AnalysisPipeline()
            .log("Loading Subject...")
            .load_subjects(path)
            .log("Trimming...")
            .trim_by_timestamp(start_time=15, end_time=250) # Keep all starting from 0 ms
            .log("Folding...")
            .fold(5)
            .log("Evaluating...")
            .evaluate_model(
                model_name=model,
                training_options={
                    "num_epochs": 20,
                    "batch_size": 64,
                    "learning_rate": 0.001,
                    "weight_decay": 0.1
                }    
            )
            .log("Saving...")
            .save(to=trained_data)
        )

        plot_roc_curve(subject=trained_data.subjects[0], filepath=f"{model}_outputs/{path.split("/")[-1]}_roc")
        plot_confusion_matrix(subject=trained_data.subjects[0], filepath=f"{model}_outputs/{path.split("/")[-1]}_conf_matrix")
        save_subject_csv(subject=trained_data.subjects[0], filepath=f"{model}_outputs/{path.split("/")[-1]}_prediction_data.csv")



