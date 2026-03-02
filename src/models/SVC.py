from .utils import ModelInterface

from sklearn.svm import SVC
import numpy as np


class SVM(ModelInterface):
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

            # pull hyperparams from training_options with sane defaults
            model = SVC(
                C=self.training_options.get("C", 1.0),
                kernel=self.training_options.get("kernel", "rbf"),
                gamma=self.training_options.get("gamma", "scale"),
                degree=self.training_options.get("degree", 3),
                probability=self.training_options.get("probability", False),
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            all_preds.extend(preds)
            all_labels.extend(y_test)

        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        return float(acc)

    def train(self):
        pass

    def infer(self):
        pass
