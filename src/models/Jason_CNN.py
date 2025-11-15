from ..core.eeg_subject import EEGSubject
from ..core.eeg_trial import EEGTrial
from ..core.ffr_prep import FFRPrep
from ..core.ffr_proc import get_accuracy

from .utils import ModelInterface

class Jason_CNN(ModelInterface):
    def __init__(self, training_options: dict[str, any]):
        super().__init__(training_options)

    def evaluate(self) -> float:
        if self.subject is None:
            raise RuntimeError(
                "No subject set. Call set_subject() before calling evaluate()."
            )

        subject: EEGSubject = self.subject
        prep = FFRPrep()
        if not subject.folds:
            subject.folds = prep.make_folds(subject, num_folds=5)

        folds = subject.folds
        for fold in folds:
            test_trials = fold
            train_trials = [t for t in subject.trials if t not in test_trials]
            self._train_on_trials(train_trials)
            predicted_labels = self.infer(test_trials)
            for trial, label in zip(test_trials, predicted_labels):
                trial.prediction.prediction = label
        
        return get_accuracy(subject)

    def train(self):
        pass

    def infer(self):
        pass