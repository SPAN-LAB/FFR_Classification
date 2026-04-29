from .utils import ModelInterface
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from ..core import EEGTrial
get_accuracy = EEGTrial.get_accuracy


class LDA(ModelInterface): 
    def __init__(self, training_options: dict[str, any]):
        super().__init__(training_options)
        
        # SVD solver will complain if variables are collinear. 
        # Using lsqr + auto shrinkage usually fixes this and is better for EEG data.
        solver = self.training_options.get("solver", "lsqr")
        shrinkage = self.training_options.get("shrinkage", "auto")
        
        self.model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    def train(self, pickle_to=None):
        #Get all trials for this subject
        X_train = np.array([t.data for t in self.subject.trials])
        y_train = np.array([t.label for t in self.subject.trials])
        self.model.fit(X_train, y_train)

        if pickle_to is not None:
            import pickle
            from pathlib import Path
            if isinstance(pickle_to, str):
                pickle_to = Path(pickle_to)
            pickle_to.parent.mkdir(parents=True, exist_ok=True)
            with open(pickle_to, 'wb') as f:
                pickle.dump(self, f)

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

        for i, fold in enumerate(folds):
            test_trials = fold
            train_trials = []
            for j, other_fold in enumerate(folds):
                if i != j:
                    train_trials.extend(other_fold)

            # Train
            X_train = np.array([t.data for t in train_trials])
            y_train = np.array([t.label for t in train_trials])
            self.model.fit(X_train, y_train)

            #Infer on test trials
            self.infer(test_trials)

            all_trials.extend(test_trials)

        #Automatically compute accuracy using helper
        return get_accuracy(self.subject.trials)