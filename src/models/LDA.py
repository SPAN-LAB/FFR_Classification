from .utils import ModelInterface
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from src.core.ffr_proc import get_accuracy
class LDA(ModelInterface):
    def __init__(self, training_options: dict[str, any]):
        super().__init__(training_options)
        self.model = LinearDiscriminantAnalysis()

    def train(self):
        #Get all trials for this subject
        X_train = np.array([t.data for t in self.subject.trials])
        y_train = np.array([t.raw_label for t in self.subject.trials])
        self.model.fit(X_train, y_train)

    def infer(self, trials):
        #Predict probabilities for given trials
        X = np.array([t.data for t in trials])
        probas = self.model.predict_proba(X)

        for trial, prob in zip(trials, probas):
            #Set prediction distribution per trial
            trial.prediction_distribution = {
                str(label): float(p)
                for label, p in zip(self.model.classes_, prob)
            }
            #Derive predicted label
            trial.prediction = self.model.classes_[np.argmax(prob)]

    def evaluate(self) -> float:
        folds = self.subject.folds
        all_trials = []

        for fold in folds:
            test_trials = fold
            train_trials = [t for t in self.subject.trials if t not in test_trials]

            # Train
            X_train = np.array([t.data for t in train_trials])
            y_train = np.array([t.raw_label for t in train_trials])
            self.model.fit(X_train, y_train)

            #Infer on test trials
            self.infer(test_trials)

            all_trials.extend(test_trials)

        #Automatically compute accuracy using helper
        return get_accuracy(self.subject)