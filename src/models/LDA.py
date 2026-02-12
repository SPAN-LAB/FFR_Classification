from .utils import ModelInterface
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

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
            trial.prediction = {
                str(label): float(p)
                for label, p in zip(self.model.classes_, prob)
            }
            #Derive predicted label
            trial.predicted_label = self.model.classes_[np.argmax(prob)]

    def evaluate(self) -> float:
        all_preds = []
        all_labels = []

        folds = self.subject.folds

        for fold in folds:
            test_trials = fold
            train_trials = [t for t in self.subject.trials if t not in test_trials]

            #Train model on training trials
            X_train = np.array([t.data for t in train_trials])
            y_train = np.array([t.raw_label for t in train_trials])
            self.model.fit(X_train, y_train)

            #Infer on test trials
            self.infer(test_trials)

            #Collect predictions
            for trial in test_trials:
                all_preds.append(trial.predicted_label)
                all_labels.append(trial.raw_label)

        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        return acc
