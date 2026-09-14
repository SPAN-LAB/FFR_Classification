import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import time
from .utils.model_interface import ModelInterface

class PureAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, input_size)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AESVM(ModelInterface):
    def __init__(self, training_options: dict[str, any]):
        super().__init__(training_options)
        self.latent_dim = self.training_options.get("latent_dim", 128)
        self.epochs = self.training_options.get("num_epochs", 20)
        self.batch_size = self.training_options.get("batch_size", 32)
        self.learning_rate = self.training_options.get("learning_rate", 1e-3)
        self.weight_decay = self.training_options.get("weight_decay", 1e-5)
        self.ae_patience = self.training_options.get("ae_patience", 8)
        self.ae_min_improvement = self.training_options.get("ae_min_improvement", 1e-4)
        self.use_svm_grid_search = self.training_options.get("use_svm_grid_search", True)
        self.svm_cv_folds = self.training_options.get("svm_cv_folds", 3)
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def _extract_xy(self, trials):
        X = np.stack([np.array(t.data, dtype=np.float32) for t in trials])
        y = np.array([int(getattr(t, "raw_label", getattr(t, "enumerated_label", 0))) for t in trials])
        return X, y

    def evaluate(self) -> float:
        if self.subject is None:
            raise RuntimeError("No subject set.")

        folds = self.subject.folds
        input_size = self.subject.trial_size
        num_folds = len(folds)

        for i, fold in enumerate(folds):
            fold_start_time = time.time()
            
            test_trials = fold
            train_trials = []
            for j in range(num_folds):
                if j != i:
                    train_trials.extend(folds[j])

            X_train_np, y_train_np = self._extract_xy(train_trials)
            X_test_np, y_test_np = self._extract_xy(test_trials)

            X_train_t = torch.tensor(X_train_np).to(self.device)
            X_test_t = torch.tensor(X_test_np).to(self.device)

            autoencoder = PureAutoencoder(input_size, self.latent_dim).to(self.device)
            optimizer = torch.optim.AdamW(
                autoencoder.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
            )
            criterion = nn.MSELoss()

            autoencoder.train()
            n_samples = X_train_t.shape[0]
            best_loss = float("inf")
            best_state = None
            no_improve = 0
            for epoch in range(1, self.epochs + 1):
                epoch_start_time = time.time()
                epoch_loss = 0.0
                steps = 0
                
                permutation = torch.randperm(n_samples)
                for j in range(0, n_samples, self.batch_size):
                    indices = permutation[j:j+self.batch_size]
                    batch_x = X_train_t[indices]
                    
                    optimizer.zero_grad()
                    recon = autoencoder(batch_x)
                    loss = criterion(recon, batch_x)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    steps += 1
                
                avg_loss = epoch_loss / steps
                scheduler.step(avg_loss)

                if avg_loss + self.ae_min_improvement < best_loss:
                    best_loss = avg_loss
                    best_state = {
                        k: v.detach().cpu() for k, v in autoencoder.state_dict().items()
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.ae_patience:
                        final_epoch_time = time.time() - epoch_start_time
                        final_fold_time = time.time() - fold_start_time
                        final_loss = avg_loss
                        break

                epoch_time = time.time() - epoch_start_time
                fold_total_time = time.time() - fold_start_time
                if epoch < self.epochs:
                    print(f"Fold [{i+1}/{num_folds}] ({fold_total_time:.1f}s total), Epoch [{epoch}/{self.epochs}] ({epoch_time:.3f}s / epoch){' ' * 20}", end="\r", flush=True)
                else:
                    final_epoch_time = epoch_time
                    final_fold_time = fold_total_time
                    final_loss = avg_loss

            if best_state is not None:
                autoencoder.load_state_dict(best_state, strict=True)
                autoencoder.to(self.device)

            autoencoder.eval()
            with torch.no_grad():
                features_train = autoencoder.encoder(X_train_t).cpu().numpy()
                features_test = autoencoder.encoder(X_test_t).cpu().numpy()

            if self.use_svm_grid_search:
                label_counts = np.bincount(y_train_np)
                min_class = int(label_counts[label_counts > 0].min()) if np.any(label_counts > 0) else 2
                cv_folds = max(2, min(self.svm_cv_folds, min_class))

                svm_pipeline = make_pipeline(
                    StandardScaler(),
                    SVC(class_weight='balanced')
                )
                param_grid = {
                    "svc__kernel": ["rbf"],
                    "svc__C": [0.5, 1.0, 5.0, 10.0],
                    "svc__gamma": ["scale", 0.01, 0.1, 1.0],
                }
                inner_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                grid = GridSearchCV(
                    estimator=svm_pipeline,
                    param_grid=param_grid,
                    cv=inner_cv,
                    scoring="accuracy",
                    n_jobs=-1,
                )
                grid.fit(features_train, y_train_np)
                predictions = grid.best_estimator_.predict(features_test)
            else:
                svm = make_pipeline(
                    StandardScaler(),
                    SVC(kernel='rbf', class_weight='balanced')
                )
                svm.fit(features_train, y_train_np)
                predictions = svm.predict(features_test)

            fold_correct = np.sum(predictions == y_test_np)
            val_acc = fold_correct / len(y_test_np)

            print(f"Fold [{i+1}/{num_folds}] ({final_fold_time:.1f}s total), Epoch [{self.epochs}/{self.epochs}] ({final_epoch_time:.3f}s / epoch), train loss={final_loss:.3f}, val acc={val_acc:.3f}")

            for trial, pred in zip(test_trials, predictions):
                trial.prediction = int(pred)

        t, s = 0, 0
        for trial in self.subject.trials:
            t += 1
            if int(getattr(trial, "raw_label", getattr(trial, "enumerated_label", 0))) == trial.prediction:
                s += 1
                
        return (s / t) 

    def train(self): pass
    def infer(self, trials): pass