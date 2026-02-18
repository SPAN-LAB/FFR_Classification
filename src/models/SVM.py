from .utils import ModelInterface
from src.core.ffr_proc import get_accuracy
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

class SVM(ModelInterface):
    """
    Support Vector Machine model implementing ModelInterface.
    Includes automatic hyperparameter search for C, gamma, and kernel.
    """

    # Default hyperparameter grid
    PARAM_GRID = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1],
        'kernel': ['rbf', 'linear']  # add 'poly' if you want to test polynomial kernel
    }

    def __init__(self, training_options: dict[str, any] = None):
        super().__init__(training_options or {})
        # Grid search options can be overridden
        self.param_grid = training_options.get('param_grid', self.PARAM_GRID) if training_options else self.PARAM_GRID
        self.model: SVC | None = None
        self.scaler: StandardScaler | None = None
        self.best_params: dict | None = None

    def train(self):
        """
        Train the SVM on all trials for this subject using GridSearchCV.
        """
        # Get training data from all trials
        X_train = np.array([t.data for t in self.subject.trials])
        y_train = np.array([t.raw_label for t in self.subject.trials])

        # Scale data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Grid search with 3-fold CV
        base_svc = SVC(probability=True, random_state=42)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid = GridSearchCV(base_svc, self.param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)

        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        print(f"[SVM] Best hyperparameters: {self.best_params}")

    def infer(self, trials):
        """
        Predict labels for the given trials.
        Stores both prediction and probability distribution.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained yet. Call train() first.")

        X = np.array([t.data for t in trials])
        X_scaled = self.scaler.transform(X)

        probas = self.model.predict_proba(X_scaled)
        for trial, prob in zip(trials, probas):
            trial.prediction_distribution = {
                str(label): float(p) for label, p in zip(self.model.classes_, prob)
            }
            trial.prediction = self.model.classes_[np.argmax(prob)]

    def evaluate(self) -> float:
        """
        Fold-based evaluation using the subject's folds.
        Returns accuracy using get_accuracy().
        """
        folds = self.subject.folds
        all_trials = []

        for fold in folds:
            test_trials = fold
            # All trials not in the current fold are used for training
            train_trials = [t for t in self.subject.trials if t not in test_trials]

            # Prepare training data
            X_train = np.array([t.data for t in train_trials])
            y_train = np.array([t.raw_label for t in train_trials])

            # Scale data
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            # Grid search for this fold
            base_svc = SVC(probability=True, random_state=42)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid = GridSearchCV(base_svc, self.param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train_scaled, y_train)

            self.model = grid.best_estimator_
            self.best_params = grid.best_params_
            print(f"[Fold Eval] Best params: {self.best_params}")

            # Predict on test fold
            self.infer(test_trials)
            all_trials.extend(test_trials)

        # Return overall accuracy
        return get_accuracy(self.subject)
