import typing
if not hasattr(typing, 'Self'):
    typing.Self = typing.Any
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from src.models.Autoencoder import _GlobalAutoencoder
from src.core.analysis_pipeline import AnalysisPipeline

def plot_true_reconstruction(autoencoder, x_raw, y_labels, device, tone_names=["Tone 1", "Tone 2", "Tone 3", "Tone 4"]):
    """Passes all 128 features together natively to get the true reconstructed waveforms."""
    autoencoder.eval()
    
    # 1. Let the AI decode the full wave naturally (all 128 features at once)
    with torch.no_grad():
        reconstructed_tensor = autoencoder(x_raw.to(device))
        reconstructed_waves = reconstructed_tensor.cpu().squeeze().numpy()
        
    labels_np = y_labels.numpy()
    
    # 2. Create the 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    fig.suptitle("True AI Reconstruction (All 128 Features Processed Together)", fontsize=16, fontweight='bold')
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    for class_label in range(4):
        # Find all brainwaves for this specific tone
        tone_indices = np.where(labels_np == class_label)[0]
        
        # Average the Reconstructed waves for this tone across all trials
        avg_recon = np.mean(reconstructed_waves[tone_indices], axis=0)
        
        ax = axes[class_label]
        
        ax.plot(avg_recon, color=colors[class_label], label=f'{tone_names[class_label]} Reconstructed', linewidth=2)
        ax.set_title(tone_names[class_label], fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if class_label >= 2: 
            ax.set_xlabel("Timepoints (0 to 4997)")
        if class_label % 2 == 0: 
            ax.set_ylabel("Amplitude")

    plt.tight_layout()
    
    # 3. Save the plot
    save_dir = "Cj/feature plots"
    os.makedirs(save_dir, exist_ok=True) 
    save_path = os.path.join(save_dir, "True_Combined_Reconstruction_Grid.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    print(f"Saved True Reconstruction plot to: {save_path}")
    plt.close()
    
def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Running Decoder Reconstruction on device: {device}")

    target_subject = "/Volumes/gurindapalli/projects/trial_classification/4tone_cell/4T1015.mat"
    
    pipeline = AnalysisPipeline()
    print("Loading data and applying subaverage(size=1)...")
    pipeline.load_subjects(target_subject)
    pipeline.subaverage(size=1)
    
    subject = pipeline.subjects[0]
    trials = subject.trials 
    
    x_raw = torch.stack([torch.tensor(trial.data, dtype=torch.float32) for trial in trials])
    y_labels = torch.tensor([trial.enumerated_label for trial in trials], dtype=torch.long)
    
    dataset = TensorDataset(x_raw, y_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Training Autoencoder...")

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

    print("\nGenerating the true simultaneous reconstruction plot...")
    plot_true_reconstruction(autoencoder, x_raw, y_labels, device)

if __name__ == "__main__":
    main()
