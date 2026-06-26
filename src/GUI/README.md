# FFR Classification GUI

A graphical user interface for the SPAN Lab FFR Classification EEG analysis toolbox.

## Quick Start

### Running the GUI

From the project root directory:

```bash
# Make sure you have dependencies installed
pip install -r requirements.txt

# Run the GUI
python run_gui.py
```

Or you can run it directly from the src/GUI folder:

```bash
python -m src.GUI.main
```

## Features

The GUI provides an intuitive interface for EEG data analysis with the following sections:

### 1. Data Loading Tab
- **Data Selection**: Browse and select your EEG data file (.mat format)
- **Data Information**: View file path and size details
- Supports single .mat files and directories containing multiple .mat files

### 2. Processing Tab
- **Trim Data**: Optionally trim data by timestamp range
  - Specify start and end times in milliseconds
  - Leave end time as infinity to keep all data
  
- **Subaverage**: Optionally apply subaveraging
  - Specify the subaveraging window size
  - Used for averaging multiple epochs together
  
- **Fold Data**: Optionally create cross-validation folds
  - Specify the number of folds
  - Useful for cross-validation experiments

*All processing steps are optional - enable only what you need*

### 3. Model Training Tab
- **Model Selection**: Choose from available models:
  - Linear Discriminant Analysis (LDA)
  - Support Vector Machine (SVM)
  - Feed-Forward Neural Network (FFNN)
  - Convolutional Neural Network (CNN)
  - Recurrent Neural Network (RNN)
  - Gated Recurrent Unit (GRU)
  - Long Short-Term Memory (LSTM)
  - Transformer
  - Jason_CNN

- **Training Parameters**:
  - **Epochs**: Number of training iterations (default: 50)
  - **Batch Size**: Samples per batch (default: 64)
  - **Learning Rate**: Optimization step size (default: 0.001)
  - **Weight Decay**: L2 regularization factor (default: 0.1)

### 4. Output Tab
- Real-time log of the analysis pipeline execution
- Status messages and error reporting
- Tracks progress through each stage

## Workflow

1. **Select Data**: Go to "Data Loading" tab and browse for your EEG data file
2. **Configure Processing** (optional): Set up any data preprocessing in the "Processing" tab
3. **Select Model & Parameters**: Choose a model and training options in the "Model Training" tab
4. **Run Analysis**: Click "Run Analysis" button to execute the pipeline
5. **Monitor Progress**: Watch the log output in the "Output" tab
6. **View Results**: Check the completion message and log for results

## Example Analysis Pipeline

A typical workflow might look like:

1. Load a 4-tone EEG dataset
2. Trim data to milliseconds 0-500
3. Apply subaverage with window size 5
4. Train an LDA classifier with default parameters
5. Review results in the output log

## System Requirements

- Python 3.11+
- PyQt5
- All dependencies from `requirements.txt`

## Supported File Formats

- **.mat files** - MATLAB format EEG data (requires pymatreader)
- Single files or directories of .mat files

## Troubleshooting

### GUI won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version is 3.11+

### Data file not loading
- Verify the file path is correct
- Ensure the file is in .mat format
- Check file permissions

### Model training fails
- Check that data was properly loaded
- Verify training parameters are reasonable
- Check available system memory for large datasets

## Tips & Best Practices

- Start with default parameters for initial exploration
- Use LDA or SVM for quick experiments
- Neural networks (CNN, LSTM) typically require longer training
- Enable subaveraging for noisy data
- Use cross-validation folds to assess model generalization
- Check the output log for detailed error messages

## Notes

- The GUI runs analysis in background threads to prevent freezing
- Large datasets may take significant time to process
- Progress is shown in the output log during execution
- All settings persist within a session (reset with "Clear All" button)

## Support

For issues or questions about the analysis pipeline, refer to the main project README.md or contact the SPAN Lab team.
