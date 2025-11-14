from .utils import ModelInterface

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np


class LDA(ModelInterface): 
    def __init__(self, training_options: dict[str, any]):
        super().__init__(training_options)

    def evaluate(self) -> float:
        subject = self.subject

        
        all_preds = []
        all_labels = []

        folds = subject.folds 

        for fold in folds:
            test_trials = fold
            train_trials = [t for t in subject.trials if t not in test_trials]

            X_train = np.array([t.data for t in train_trials])
            y_train = np.array([t.raw_label for t in train_trials])

            X_test = np.array([t.data for t in test_trials])
            y_test = np.array([t.raw_label for t in test_trials])

            model = LinearDiscriminantAnalysis()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            all_preds.extend(preds)
            all_labels.extend(y_test)

        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        return acc

    def train(self):
        pass

    def infer(self):
        pass