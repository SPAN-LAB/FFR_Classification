from typing import Any, Callable
from abc import ABC, abstractmethod

from EEGTrial import EEGTrial, EEGTrialInterface
from EEGSubject import EEGSubject, EEGSubjectInterface
import matplotlib.pyplot as plt
import seaborn as sns

class PlotsInterface(ABC): 

    @abstractmethod
    @staticmethod
    def plot_single_trial(trial: EEGTrialInterface): ...

    @abstractmethod
    @staticmethod
    def plot_averaged_trials(trials: list[EEGTrialInterface], show_components: bool, key: Callable[[EEGTrialInterface], Any]): ...

    @abstractmethod
    @staticmethod
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
        return ax 

    @staticmethod
    def plot_averaged_trials(subject: EEGSubject, show_components: bool=True, key: Callable[[EEGTrial], Any]=lambda trial: trial.mapped_label): 
        # Group trials and for each group compute a single subaverage by collating all trials
        grouped = subject.grouped_trials(key=key)

        fig, ax = plt.subplots()

        for group_key, trials in grouped.items():
            # Optionally plot individual trial components with low alpha
            if show_components:
                for t in trials:
                    sns.lineplot(x=t.timestamps, y=t.data, ax=ax, alpha=0.15, linewidth=1, legend=False)

            # Create a pseudo-subject containing only this homogeneous group
            pseudo_subject = EEGSubject(trials=trials)
            pseudo_subject.subaverage(size=len(trials))

            # After subaverage of full group size, expect exactly one averaged trial
            if not pseudo_subject.trials:
                continue
            averaged_trial = pseudo_subject.trials[0]

            sns.lineplot(x=averaged_trial.timestamps, y=averaged_trial.data, ax=ax, label=str(group_key))

        ax.set_xlabel("Time")
        ax.set_ylabel("Signal")
        ax.set_title("Averaged Trials")
        ax.legend()
        return ax

    @staticmethod
    def plot_grand_average(subjects: list[EEGSubject], show_components: bool=True):
        pass