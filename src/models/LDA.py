from torch import nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
import numpy as np

class LDA_Model(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.lda = LinearDiscriminantAnalysis()
        self.num_classes = num_classes

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.lda.fit(X, y)

    def forward(self, X: np.ndarray):
        return self.lda.predict_proba(X)

class LDAModelWrapper: 
    def __init__(self, subjects):
        """
        subjects: list of EEGSubject
        """
        self.subjects = subjects

    def evaluate(self) -> list[float]:
        accuracies: list[float] = []

        for subject in self.subjects:
            all_preds = []
            all_labels = []

            folds = subject.fold()  

            for fold in folds:
                test_trials = fold
                train_trials = [t for t in subject.trials if t not in test_trials]

                X_train = np.array([t.data for t in train_trials])
                y_train = np.array([t.label for t in train_trials])

                X_test = np.array([t.data for t in test_trials])
                y_test = np.array([t.label for t in test_trials])

                model = LinearDiscriminantAnalysis()
                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                all_preds.extend(preds)
                all_labels.extend(y_test)

            acc = np.mean(np.array(all_preds) == np.array(all_labels))
            accuracies.append(acc)

        return accuracies
