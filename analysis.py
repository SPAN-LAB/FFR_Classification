import numpy as np

from src.core import AnalysisPipeline
from src.core.plots import plot_confusion_matrix, plot_roc_curve

from src.analysis import accuracy_against_subaverage_size
from src.analysis import accuracy_against_data_amount
from src.analysis.utils import get_mats

from local.constants import *

# investigate_subaverage_size_vs_accuracy(
#     subaverage_sizes=list(range(2, 25, 2)), # You can use [1, 5, 10, 50], for example
#     subject_filepaths=[get_mats("< ...enter your filepath here... >")],
#     model_names=["FFNN", "CNN"],
#     training_options=TO,
#     output_folder_path="analysis_outputs"
# )

accuracy_against_data_amount(
    min_trials=20,
    stride=20,
    subject_filepaths=[GOOD_D_PATH],
    model_names=["FFNN"],
    training_options=TO256,
    output_folder_path="analysis_outputs/accuracy_against_data_amount"
)

# base = AnalysisPipeline().load_subjects(BAD_D_PATH)

# for s in [1,5,10,20,50,100]:
#     z = AnalysisPipeline()
#     base.save(to=z)
#     p = (
#         z
#         .trim_by_timestamp(start_time=50, end_time=250)
#         # .subaverage()
#         .fold()
#         .evaluate_model(
#             model_name="FFNN",
#             training_options=TO256
#         )
#     )
    
#     # plot_confusion_matrix(subject=p.subjects[0], filepath=f"analysis_outputs/test/{p.subjects[0].name}-{s}")
#     plot_roc_curve(subject=p.subjects[0], filepath=f"analysis_outputs/test/{p.subjects[0].name}/{s}.svg")