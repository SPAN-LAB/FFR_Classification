from .utils import ModelInterface
from src.core.ffr_proc import get_accuracy
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform


class SVM(ModelInterface):

    # -------------------------------
    # Grid Search Parameters (default)
    # -------------------------------
    # Note: This is based on the implementation by Jivesh
    PARAM_GRID = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }

    # -------------------------------
    # Random Search Distributions (low trade off)
    # -------------------------------
    PARAM_DIST = {
        'C': loguniform(1e-3, 1e3),
        'gamma': loguniform(1e-4, 1e1),
        'kernel': ['rbf', 'linear']
    }

    def __init__(self, training_options: dict[str, any] = None):
        super().__init__(training_options or {})

        self.search_type = self.training_options.get("search_type", "grid")
        self.n_iter = self.training_options.get("n_iter", 20)

        self.param_grid = self.PARAM_GRID
        self.param_dist = self.PARAM_DIST

        self.model: SVC | None = None
        self.scaler: StandardScaler | None = None
        self.best_params: dict | None = None

    # In demo.py/analysis.py 
    # "search_type": "random",  # <--- add this line in moldle (In TRAINING_OPTIONS in Analysis.py)
    # "n_iter": 30              # optional: number of random search iterations

    def _build_search(self, base_svc, cv):

        if self.search_type == "random":
            print("[SVM] Using RandomizedSearchCV")
            return RandomizedSearchCV(
                estimator=base_svc,
                param_distributions=self.param_dist,
                n_iter=self.n_iter,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42
            )

        else:
            print("[SVM] Using GridSearchCV")
            return GridSearchCV(
                estimator=base_svc,
                param_grid=self.param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1
            )

    def train(self):

        X_train = np.array([t.data for t in self.subject.trials])
        y_train = np.array([t.raw_label for t in self.subject.trials])

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        base_svc = SVC(probability=True, random_state=42)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        search = self._build_search(base_svc, cv)
        search.fit(X_train_scaled, y_train)

        self.model = search.best_estimator_
        self.best_params = search.best_params_

        print(f"[SVM] Best hyperparameters: {self.best_params}")


    def infer(self, trials):

        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained yet. Call train() first.")

        X = np.array([t.data for t in trials])
        X_scaled = self.scaler.transform(X)

        probas = self.model.predict_proba(X_scaled)

        for trial, prob in zip(trials, probas):
            trial.prediction_distribution = {
                str(label): float(p)
                for label, p in zip(self.model.classes_, prob)
            }
            trial.prediction = self.model.classes_[np.argmax(prob)]


    def evaluate(self) -> float:

        folds = self.subject.folds

        for fold in folds:

            test_trials = fold
            train_trials = [t for t in self.subject.trials if t not in test_trials]

            X_train = np.array([t.data for t in train_trials])
            y_train = np.array([t.raw_label for t in train_trials])

            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            base_svc = SVC(probability=True, random_state=42)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            search = self._build_search(base_svc, cv)
            search.fit(X_train_scaled, y_train)

            self.model = search.best_estimator_
            self.best_params = search.best_params_

            self.infer(test_trials)

        return get_accuracy(self.subject)