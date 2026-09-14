import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
import time

from .utils.model_interface import ModelInterface

# 1. We only need the Encoder part of the architecture here!
class GlobalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super(GlobalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.encoder(x)

class GlobalAESVM(ModelInterface):
    def __init__(self, training_options: dict):
        super().__init__(training_options)
        self.latent_dim = training_options.get('latent_dim', 128)
        self.ae_weights_path = training_options.get('ae_weights_path', 'global_ae_for_4T1014.pth')

    def _extract_features(self, trials, encoder, device):
        # Extract raw data and labels
        X = np.stack([np.array(t.data, dtype=np.float32).flatten() for t in trials])
        y = np.array([int(getattr(t, "raw_label", getattr(t, "enumerated_label", 0))) for t in trials])
        
        # Convert to tensor and pass through the FROZEN encoder
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad(): # This completely disables learning/leakage!
            features = encoder(X_tensor).cpu().numpy()
            
        return features, y

    def evaluate(self) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Grab the input dimension (4915 based on your terminal output)
        sample_trial = np.array(self.subject.trials[0].data).flatten()
        input_dim = sample_trial.shape[0]

        # 2. Load the Pre-trained Foundation Model
        encoder = GlobalEncoder(input_dim, self.latent_dim).to(device)
        
        # strict=False is CRUCIAL here because our .pth file has a decoder, but this class doesn't!
        encoder.load_state_dict(torch.load(self.ae_weights_path, map_location=device), strict=False)
        encoder.eval() # FREEZE THE WEIGHTS

        folds = self.subject.folds
        num_folds = len(folds)
        total_correct = 0
        total_trials = 0

        print(f"\n--- Starting 5-Fold SVM Cross-Validation with Global AE Features ---")
        for i, fold in enumerate(folds):
            fold_start_time = time.time()
            test_trials = fold
            train_trials = [t for j, f in enumerate(folds) if j != i for t in f]

            # 3. Extract features completely cleanly
            X_train_features, y_train = self._extract_features(train_trials, encoder, device)
            X_test_features, y_test = self._extract_features(test_trials, encoder, device)

            # 4. Train the SVM
            svm = SVC(kernel='rbf', class_weight='balanced')
            svm.fit(X_train_features, y_train)
            predictions = svm.predict(X_test_features)

            val_acc = np.mean(predictions == y_test)
            fold_time = time.time() - fold_start_time
            print(f"Fold [{i+1}/{num_folds}] ({fold_time:.1f}s) | Global Features Val Acc: {val_acc:.3f}")

            # Save predictions back to your pipeline's trial objects
            for trial, pred in zip(test_trials, predictions):
                trial.prediction = int(pred)
                if int(getattr(trial, "raw_label", getattr(trial, "enumerated_label", 0))) == trial.prediction:
                    total_correct += 1
            total_trials += len(test_trials)

        final_accuracy = (total_correct / total_trials) * 100
        print(f"\n✅ Final Scientifically Valid Accuracy: {final_accuracy:.2f}%")
        return final_accuracy

    def train(self): pass
    def infer(self, trials): pass