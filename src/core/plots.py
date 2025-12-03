from typing import Any, Callable
from abc import ABC, abstractmethod

from .eeg_trial import EEGTrial
from .eeg_subject import EEGSubject
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import ceil

def plot_single_trial(trial: EEGTrial):
    # Create a simple line plot of timestamps (x) vs data (y)
    fig, ax = plt.subplots()
    sns.lineplot(x=trial.timestamps, y=trial.data, ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    title_label = f"{trial.mapped_label}" if getattr(trial, "mapped_label", None) is not None else ""
    ax.set_title(f"Trial {trial.trial_index} {f'({title_label})' if title_label else ''}")
    try:
        fig.tight_layout()
        fig.canvas.draw_idle()
        plt.show(block=False)
    except Exception:
        pass
    return ax

def plot_averaged_trials(subject: EEGSubject, key: Callable[[EEGTrial], Any]=lambda trial: trial.raw_label):
    show_components = False
    # Group trials by key (label) and plot each label in its own subplot (2 x n grid)
    grouped = subject.grouped_trials(key=key)

    keys = list(grouped.keys())
    if not keys:
        fig, ax = plt.subplots()
        ax.set_title("No trials to plot")
        try:
            fig.tight_layout(); plt.show(block=False)
        except Exception:
            pass
        return ax

    try:
        keys = sorted(keys)
    except Exception:
        pass

    num_groups = len(keys)
    cols = ceil(num_groups / 2)

    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 6), squeeze=False)

    # Compute bounds from the per-label averaged trial only (your rule)
    avg_trials: dict[Any, EEGTrial] = {}
    label_mins = []
    label_maxs = []
    for group_key, trials in grouped.items():
        pseudo_subject = EEGSubject(trials=trials)
        pseudo_subject.subaverage(size=len(trials))
        if pseudo_subject.trials:
            averaged_trial = pseudo_subject.trials[0]
            avg_trials[group_key] = averaged_trial
            try:
                label_mins.append(float(np.nanmin(averaged_trial.data)))
                label_maxs.append(float(np.nanmax(averaged_trial.data)))
            except Exception:
                try:
                    if len(averaged_trial.data) > 0:
                        label_mins.append(float(min(averaged_trial.data)))
                        label_maxs.append(float(max(averaged_trial.data)))
                except Exception:
                    pass

    if not label_mins or not label_maxs:
        y_lo, y_hi = 0.0, 1.0
    else:
        # Minimal common range that fits all averaged traces
        y_lo = min(label_mins)
        y_hi = max(label_maxs)
        if y_hi == y_lo:
            eps = 1e-6
            y_lo -= eps
            y_hi += eps

    # Helper to retrieve target axes
    def get_ax(idx: int):
        row = 0 if idx < cols else 1
        col = idx % cols
        return axes[row][col]

    for idx, group_key in enumerate(keys):
        ax = get_ax(idx)
        trials = grouped[group_key]

        # Optionally plot individual trial components with low alpha
        if show_components:
            for t in trials:
                sns.lineplot(x=t.timestamps, y=t.data, ax=ax, color="#999999", alpha=0.35, linewidth=1.0, legend=False)

        # Use precomputed averaged trial for this label
        averaged_trial = avg_trials.get(group_key)
        if averaged_trial is not None:
            sns.lineplot(x=averaged_trial.timestamps, y=averaged_trial.data, ax=ax, color="#1f77b4", linewidth=2.5, label="avg")

        ax.set_title(f"Label {group_key}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Signal")
        ax.legend()
        ax.set_ylim(y_lo, y_hi)

    # Hide any unused subplots
    total_slots = 2 * cols
    for empty_idx in range(num_groups, total_slots):
        ax = get_ax(empty_idx)
        ax.axis('off')

    fig.suptitle("Averaged Trials by Label", y=0.98)
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.canvas.draw_idle()
        plt.show(block=False)
    except Exception:
        pass

    # Return the first axes for compatibility
    return axes[0][0]

def plot_grand_average(subjects: list[EEGSubject], show_components: bool=True):
    # Combine trials from all subjects per label
    def get_label(tr):
        return tr.mapped_label if getattr(tr, 'mapped_label', None) is not None else tr.raw_label

    # Distinct labels across all subjects
    labels_set = set()
    for subj in subjects:
        for tr in getattr(subj, 'trials', []) or []:
            labels_set.add(get_label(tr))

    keys = list(labels_set)
    if not keys:
        fig, ax = plt.subplots()
        ax.set_title("No trials to plot (grand average)")
        try:
            fig.tight_layout(); plt.show(block=False)
        except Exception:
            pass
        return ax

    try:
        keys = sorted(keys)
    except Exception:
        pass

    num_groups = len(keys)
    cols = ceil(num_groups / 2)
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 6), squeeze=False)

    # For each label, compute one per-subject averaged trace (component), then a grand average across subjects
    label_to_subject_avgs: dict[Any, list[tuple]] = {}
    label_to_grand_avg: dict[Any, tuple] = {}

    # Collect all subject-level averages to determine global y-limits
    all_series_mins = []
    all_series_maxs = []

    for lbl in keys:
        per_subject_avgs = []  # list of (timestamps, data, name)
        for idx, subj in enumerate(subjects):
            trials_lbl = [tr for tr in (getattr(subj, 'trials', []) or []) if get_label(tr) == lbl]
            if not trials_lbl:
                continue
            try:
                stack = np.stack([tr.data for tr in trials_lbl], axis=0).astype(float)
                data_avg = stack.mean(axis=0)
            except Exception:
                # Fallback: single trial
                data_avg = np.asarray(trials_lbl[0].data, dtype=float)
            ts = trials_lbl[0].timestamps
            name = f"subj {idx+1}"
            per_subject_avgs.append((ts, data_avg, name))
            try:
                all_series_mins.append(float(np.nanmin(data_avg)))
                all_series_maxs.append(float(np.nanmax(data_avg)))
            except Exception:
                if len(data_avg) > 0:
                    all_series_mins.append(float(min(data_avg)))
                    all_series_maxs.append(float(max(data_avg)))

        # Compute grand average across subjects (of their per-subject averages)
        if per_subject_avgs:
            try:
                gstack = np.stack([d for (_, d, _) in per_subject_avgs], axis=0)
                grand = gstack.mean(axis=0)
            except Exception:
                grand = per_subject_avgs[0][1]
            ts0 = per_subject_avgs[0][0]
            label_to_grand_avg[lbl] = (ts0, grand)
            try:
                all_series_mins.append(float(np.nanmin(grand)))
                all_series_maxs.append(float(np.nanmax(grand)))
            except Exception:
                if len(grand) > 0:
                    all_series_mins.append(float(min(grand)))
                    all_series_maxs.append(float(max(grand)))
        else:
            label_to_grand_avg[lbl] = None

        label_to_subject_avgs[lbl] = per_subject_avgs

    # Determine common y-limits across all series (components + grand averages)
    if not all_series_mins or not all_series_maxs:
        y_lo, y_hi = 0.0, 1.0
    else:
        y_lo = min(all_series_mins)
        y_hi = max(all_series_maxs)
        if y_hi == y_lo:
            eps = 1e-6
            y_lo -= eps
            y_hi += eps

    # Helper to retrieve target axes
    def get_ax(idx: int):
        row = 0 if idx < cols else 1
        col = idx % cols
        return axes[row][col]

    # Plot per-label components (one per subject) and grand average
    for idx, lbl in enumerate(keys):
        ax = get_ax(idx)
        per_subject_avgs = label_to_subject_avgs.get(lbl, [])
        if show_components:
            for ts, data, _name in per_subject_avgs:
                sns.lineplot(x=ts, y=data, ax=ax, color="#999999", alpha=0.35, linewidth=1.0, legend=False)

        grand = label_to_grand_avg.get(lbl)
        if grand is not None:
            gts, gdata = grand
            sns.lineplot(x=gts, y=gdata, ax=ax, color="#1f77b4", linewidth=2.75, label="grand")

        ax.set_title(f"Label {lbl}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Signal")
        ax.set_ylim(y_lo, y_hi)
        ax.legend()

    # Hide any unused subplots
    total_slots = 2 * cols
    for empty_idx in range(num_groups, total_slots):
        ax = get_ax(empty_idx)
        ax.axis('off')

    fig.suptitle("Grand Averaged Trials by Label", y=0.98)
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.canvas.draw_idle()
        plt.show(block=False)
    except Exception:
        pass

    return axes[0][0]

def plot_roc_curve(
    *, 
    subject: EEGSubject, 
    filepath: str | None = None, 
    enforce_saturation: bool = False,
    show_popup: bool = False
):
    """
    Given a list of trials, plots an ROC curve using trial.prediction_distribution
    """

    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Determine the classes from prediction_distribution keys (not trial.label)
    # This ensures type consistency
    labels_set = set()
    for trial in subject.trials:
        if trial.prediction_distribution is not None and isinstance(trial.prediction_distribution, dict):
            labels_set.update(trial.prediction_distribution.keys())
    
    if not labels_set:
        # Fallback: use trial.label if no prediction_distribution found
        for trial in subject.trials:
            labels_set.add(trial.label)
    
    labels = sorted(list(labels_set))
    n_classes = len(labels)
    
    # Collect true labels and prediction distributions
    y_true = []
    y_scores = []
    
    for trial in subject.trials:
        if enforce_saturation and (trial.prediction is None or trial.prediction_distribution is None):
            raise ValueError("Expected predictions and prediction distributions but found None.")
        
        if trial.prediction_distribution is not None:
            # Extract probabilities in the correct order based on sorted labels
            if isinstance(trial.prediction_distribution, dict):
                # Get the type of keys in prediction_distribution
                sample_key = next(iter(trial.prediction_distribution.keys()))
                
                # Convert trial.label to match the type of prediction_distribution keys
                if isinstance(sample_key, int):
                    true_label = int(trial.label)
                elif isinstance(sample_key, float):
                    true_label = float(trial.label)
                else:
                    true_label = trial.label
                
                y_true.append(true_label)
                
                # Convert dict to array in the correct order
                prob_array = [trial.prediction_distribution.get(label, 0.0) for label in labels]
                y_scores.append(prob_array)
            else:
                # Already an array/list
                y_true.append(trial.label)
                y_scores.append(trial.prediction_distribution)
    
    if not y_true:
        raise ValueError("No trials with prediction distributions found.")
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Binarize the labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=labels)
    
    # If binary classification, reshape
    if n_classes == 2:
        y_true_bin = y_true_bin.ravel()
        y_scores = y_scores[:, 1]  # Use probability of positive class
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Set seaborn style for beautiful plots
    sns.set_palette("husl")
    
    # Compute ROC curve and AUC for each class
    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        # Multi-class: one curve per class
        colors = sns.color_palette("husl", n_classes)
        for i, (label, color) in enumerate(zip(labels, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, color=color, 
                    label=f'Class {label} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier', alpha=0.5)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve, {subject.name}', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    if filepath is not None:
        # Create the necessary folders if they don't exist in the filepath
        from pathlib import Path
        filepath_obj = Path(filepath)
        if not filepath_obj.parent.exists():
            print(f"Creating directory: {filepath_obj.parent}")
            filepath_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        plt.savefig(filepath)
    
    # Show plot if requested
    if show_popup:
        plt.show()
    
    plt.close()

def plot_confusion_matrix(
    *, 
    subject: EEGSubject,
    filepath: str | None = None, 
    enforce_saturation: bool = False,
    show_popup: bool = False
):
    """
    Given a list of trials, plots a confusion matrix 
    """
    
    # Determine the classes
    labels_set = set()  
    for trial in subject.trials:
        labels_set.add(trial.label)
    labels = []
    for label in labels_set:
        labels.append(label)
    labels.sort()
    
    # Initialize the matrix 
    matrix = []
    for label in labels:
        matrix.append([0 for _ in labels])
    
    for trial in subject.trials:
        if enforce_saturation and trial.prediction is None:
            raise ValueError("Expected a trial but found None.")
        else:
            matrix[trial.enumerated_label][subject.labels_map[trial.prediction]] += 1
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,           # Show counts in cells
        fmt='d',              # Integer format
        cmap='Blues',         # Color scheme
        xticklabels=labels,   # Use actual labels on x-axis
        yticklabels=labels,   # Use actual labels on y-axis
        cbar=True             # Show colorbar
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix, {subject.name}', fontsize=14)
    plt.tight_layout()
    
    if filepath is not None:
        # Create the necessary folders if they don't exist in the filepath
        from pathlib import Path
        filepath_obj = Path(filepath)
        if not filepath_obj.parent.exists():
            print(f"Creating directory: {filepath_obj.parent}")
            filepath_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        plt.savefig(filepath)
    
    # Show plot if requested
    if show_popup:
        plt.show()
    
    plt.close()