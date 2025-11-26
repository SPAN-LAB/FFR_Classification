from typing import Any, Callable
from abc import ABC, abstractmethod

from .eeg_trial import EEGTrial, EEGTrialInterface
from .eeg_subject import EEGSubject, EEGSubjectInterface
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import ceil

class PlotsInterface(ABC):

    @staticmethod
    @abstractmethod
    def plot_single_trial(trial: EEGTrialInterface): ...

    @staticmethod
    @abstractmethod
    def plot_averaged_trials(trials: list[EEGTrialInterface], show_components: bool, key: Callable[[EEGTrialInterface], Any]): ...

    @staticmethod
    @abstractmethod
    def plot_grand_average(subjects: list[EEGSubjectInterface], show_components: bool): ...

class Plots(PlotsInterface):

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
