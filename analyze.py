"""
SPAN Lab - FFR Classification

Filename: analyze.py
Author(s): Kevin Chen
Description: TODO
"""


from src.analysis import subaverage_size, data_amount
import argparse


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
generic        = args.generic

# Decide which type of analysis to run
if generic == "true":
    from local.constants import ALL_D_PATHS
    if analysis_type == "data_amount":
        data_amount.generic_analyze(
            model_name=model_name,
            subject_filepath=data_filepath,
            all_subject_filepaths=ALL_D_PATHS,
            output_dirpath=output_dirpath
        )
    elif analysis_type == "subaverage_size":
        subaverage_size.generic_analyze(
            model_name=model_name,
            subject_filepath=data_filepath,
            all_subject_filepaths=ALL_D_PATHS,
            output_dirpath=output_dirpath
        )
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")
else:
    if analysis_type == "subaverage_size":
        subaverage_size.analyze(
            model_name=model_name,
            subject_filepath=data_filepath,
            output_dirpath=output_dirpath
        )
    elif analysis_type == "data_amount":
        data_amount.analyze(
            model_name=model_name,
            subject_filepath=data_filepath,
            output_dirpath=output_dirpath
        )
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")
