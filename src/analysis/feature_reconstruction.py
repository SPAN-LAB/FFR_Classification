import typing
if not hasattr(typing, 'Self'):
    typing.Self = typing.Any
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt

from src.models.Autoencoder import _GlobalAutoencoder
from src.core.analysis_pipeline import AnalysisPipeline

def isolate_and_plot_feature(autoencoder, x_raw, y_labels, feature_idx, device, tone_names=["Tone 1", "Tone 2", "Tone 3", "Tone 4"]):
    """Mutes 127 features, decodes the 'solo track' of 1 feature, and averages by tone."""
    autoencoder.eval()
    
    with torch.no_grad():
        original_features = autoencoder.encoder(x_raw.to(device))
        solo_features = torch.zeros_like(original_features)
        solo_features[:, feature_idx] = original_features[:, feature_idx]
        reconstructed_waves = autoencoder.decoder(solo_features).cpu().squeeze().numpy()

    labels_np = y_labels.numpy()
    plt.figure(figsize=(12, 6))
    
    for class_label in range(4):
        tone_indices = np.where(labels_np == class_label)[0]
        tone_waves = reconstructed_waves[tone_indices]
        averaged_wave = np.mean(tone_waves, axis=0)
        plt.plot(averaged_wave, label=f"{tone_names[class_label]}")

    plt.title(f"Pure Physical Shape of Latent Feature #{feature_idx}")
    plt.xlabel("Timepoints (0 to 4997)")
    plt.ylabel("Amplitude (Reconstructed)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # --- NEW AUTOMATIC SAVING LOGIC ---
    save_dir = "Cj/feature plots"
    os.makedirs(save_dir, exist_ok=True) # Creates the folder if it doesn't exist
    
    filename = f"Feature_{feature_idx}_Reconstruction.png"
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # Saves high-res image
    print(f"Saved highly detailed plot to: {save_path}")
    
    plt.close() # Closes the figure to free up your computer's memory

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Running Decoder Reconstruction on device: {device}")

    target_subject = "your_path"
    
    pipeline = AnalysisPipeline()
    print("Loading data and applying subaverage(size=1)...") # Fixed this print statement!
    pipeline.load_subjects(target_subject)
    pipeline.subaverage(size=1)
    
    subject = pipeline.subjects[0]
    trials = subject.trials 
    
    x_raw = torch.stack([torch.tensor(trial.data, dtype=torch.float32) for trial in trials])
    y_labels = torch.tensor([trial.enumerated_label for trial in trials], dtype=torch.long)
    
    dataset = TensorDataset(x_raw, y_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Training Autoencoder to get the 128 Latent Features...")

    autoencoder = _GlobalAutoencoder(
        input_dim=x_raw.shape[-1],
        latent_dim=128,
    ).to(device)
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    ae_criterion = nn.MSELoss()

    autoencoder.train()
    for epoch in range(30): 
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            ae_optimizer.zero_grad()
            reconstruction = autoencoder(batch_x)
            loss = ae_criterion(reconstruction.squeeze(), batch_x.squeeze())
            loss.backward()
            ae_optimizer.step()

    autoencoder.eval()
    with torch.no_grad():
        features_cpu = autoencoder.encoder(x_raw.to(device)).cpu()
        
    labels_cpu = y_labels.cpu()
    X_train, X_test, y_train, y_test = train_test_split(
        features_cpu, labels_cpu, test_size=0.2, random_state=42, stratify=labels_cpu
    )
    
    print("\nTraining SVM to find the Top 3 Most Important Features...")
    svm_clf = SVC(kernel='rbf', class_weight='balanced')
    svm_clf.fit(X_train.numpy(), y_train.numpy())
    
    result = permutation_importance(
        svm_clf, X_test.numpy(), y_test.numpy(), n_repeats=10, random_state=42, n_jobs=-1
    )

    top_3_indices = result.importances_mean.argsort()[-3:][::-1]
    print(f"The Top 3 VIP Features are: {top_3_indices}")

    for feature_idx in top_3_indices:
        print(f"\nDecoding the 'Solo Track' for Feature #{feature_idx}...")
        isolate_and_plot_feature(autoencoder, x_raw, y_labels, feature_idx, device)

if __name__ == "__main__":
    main()
