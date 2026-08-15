import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from ..core import AnalysisPipeline
from ..models.utils import find_model

def analyze_saliency_phase_locking(
    subject_filepaths: list[str],
    model_name: str,
    training_options: dict,
    output_folder_path: str,
    target_tones: list[int] = [1, 2, 3, 4],
    sampling_rate: int = 16384,
    subaverage_size: int = 1,
    num_folds: int = 5  # Added parameter for folding
):
    """
    Trains a model on the provided subjects and then performs Spectral Analysis 
    on the model's saliency maps.
    """
    
    print(f"--- Starting Saliency Phase-Locking Analysis ---")
    
    # 1. Initialize Pipeline & Load Data
    pipeline = AnalysisPipeline()
    pipeline.load_subjects(subject_filepaths)
    
    # 2. Pre-processing (CRITICAL FIX)
    # The model expects data to be split into folds for evaluation.
    if subaverage_size > 1:
        pipeline.subaverage(subaverage_size)
    
    print(f"Folding data into {num_folds} folds...")
    pipeline.fold(num_folds)  # <--- THIS WAS MISSING
    
    # 3. Train the Model
    print(f"Training {model_name}...")
    pipeline.evaluate_model(model_name, training_options)

    # Ensure output directory exists
    os.makedirs(output_folder_path, exist_ok=True)

    # 4. Analyze Each Subject
    for i, subject in enumerate(pipeline.subjects):
        # Safety Check: If training failed, skip this subject to avoid IndexError
        if i >= len(pipeline.models):
            print(f"  No model found for subject {subject.name}. Training likely failed.")
            continue

        model_wrapper = pipeline.models[i]
        if getattr(model_wrapper, "required_inputs", ["raw"]) != ["raw"]:
            print(f"  {model_name} requires non-raw inputs; saliency currently supports raw-only models.")
            continue
        
        # Access the internal PyTorch module
        if hasattr(model_wrapper, 'model'):
            torch_model = model_wrapper.model
        elif hasattr(model_wrapper, 'network'):
            torch_model = model_wrapper.network
        else:
            print(f" Could not find internal PyTorch model for {subject.name}. Skipping.")
            continue
            
        torch_model.eval()
        print(f"\nAnalyzing Subject: {subject.name}")

        # Loop through Targets (Tones)
        for target_tone in target_tones:
            # A. Prepare Data: Compute Grand Average for this tone
            tone_trials = [t for t in subject.trials if t.raw_label == target_tone]
            
            if not tone_trials:
                print(f"  No trials found for Tone {target_tone}")
                continue
            
            # Stack data: (N_trials, Time)
            X_np = np.stack([t.data for t in tone_trials])
            grand_avg = np.mean(X_np, axis=0)
            
            # Prepare Tensor: (1, 1, Time)
            device = next(torch_model.parameters()).device
            input_tensor = (
                torch.tensor(grand_avg, dtype=torch.float32, device=device)
                .reshape(1, 1, -1)
                .clone()
                .detach()
                .requires_grad_(True)
            )
            
            # B. Forward Pass & Backprop
            torch_model.zero_grad()
            output = torch_model(input_tensor)
            
            target_idx = subject.labels_map.get(target_tone)
            if target_idx is None:
                print(f"  Warning: Tone {target_tone} is not in the subject label map")
                continue
            if output.shape[1] <= target_idx:
                print(f"  Warning: Output size {output.shape} too small for target {target_idx}")
                continue

            score = output[0, target_idx]
            score.backward()
            
            # Saliency is the gradient magnitude
            saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
            
            # C. FFT of Saliency
            N = len(saliency)  # <--- Defined N here
            T = 1.0 / sampling_rate
            
            yf = fft(saliency)
            xf = fftfreq(N, T)
            
            # Filter frequencies (20-1000Hz)
            mask = (xf > 20) & (xf < 1000)
            freqs = xf[mask]
            power = np.abs(yf[mask])
            
            # Normalize
            if np.max(power) > 0:
                power = power / np.max(power)
            
            # D. Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(freqs, power, color='blue', linewidth=2)
            
            peak_idx = np.argmax(power)
            peak_freq = freqs[peak_idx]
            
            plt.title(f"Saliency FFT - Tone {target_tone} - {subject.name}\n(Peak: {peak_freq:.1f} Hz)", fontsize=14)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Attention Power (Normalized)")
            plt.axvline(peak_freq, color='red', linestyle='--', alpha=0.7, label=f"Peak: {peak_freq:.1f} Hz")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_name = f"{subject.name}_Tone{target_tone}_SaliencyFFT.png"
            save_path = os.path.join(output_folder_path, save_name)
            plt.savefig(save_path)
            plt.close()
            
            print(f"  Tone {target_tone}: Peak at {peak_freq:.1f} Hz -> Saved")

    print("\nAnalysis Complete.")
