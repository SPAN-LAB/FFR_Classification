"""
SPAN Lab - FFR Classification

Filename: analyze.py
Author(s): Kevin Chen
Description: TODO
"""


import argparse
from src.analysis.data_amount import analyze2, AnalysisType
from local.constants import ALL_D_PATHS


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--analysis_type",  type=str, required=True)
parser.add_argument("--model_name",     type=str, required=True)
parser.add_argument("--data_filepath",  type=str, required=True)
parser.add_argument("--output_dirpath", type=str, required=True)
parser.add_argument("--generic",        type=str, required=True)
args = parser.parse_args()
analysis_type  = args.analysis_type
model_name     = args.model_name
data_filepath  = args.data_filepath
output_dirpath = args.output_dirpath
generic        = args.generic == "true"

if analysis_type == AnalysisType.DATA_AMOUNT:
    data_amounts = list(range(
        200,      # Smallest data amount
        3800 + 1, # Largest data amount
        100       # Stride
    ))
    subaverage_sizes = [5]
elif analysis_type == AnalysisType.SUBAVERAGE_SIZE:
    data_amounts = []
    subaverage_sizes = [
          1,   5,  10,  15,  20,
         25,  30,  35,  40,  45,
         50,  55,  60,  65,  70,
         75,  80,  85,  90,  95,
        100, 105, 110, 115, 120
    ]

if generic:
    all_subject_filepaths = ALL_D_PATHS
else:
    all_subject_filepaths = []

analyze2(
    analysis_type=analysis_type,
    is_generic=generic,
    model_name=model_name,
    subaverage_sizes=subaverage_sizes,
    data_amounts=data_amounts,
    subject_filepath=data_filepath,
    all_subject_filepaths=all_subject_filepaths,
    output_root_dirpath=output_dirpath
)
