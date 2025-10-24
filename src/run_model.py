import importlib
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def get_device():
    """Auto-detect best available device across all systems."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
@torch.no_grad()
def run_model(
    *,
    model_name: str,
    model_kwargs: Dict,
    dataloader: DataLoader,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict:
    mod = importlib.import_module(f"src.models.{model_name}")
    ModelClass = getattr(mod, "Model")
    model: nn.Module = ModelClass(**model_kwargs)

    if device is None:
        device = get_device()
    print(f"Using device: {device}")

    model.to(device)
    if weights_path:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=True)
    model.eval()

    total_n, correct = 0, 0
    for batch in dataloader:
        if len(batch) < 2:
            raise ValueError("Each batch must be at least (x, y).")
        x, y = batch[0].to(device), batch[1].to(device)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Expected x with 2 or 3 dims, got {tuple(x.shape)}")
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total_n += y.numel()

    acc = (correct / total_n) if total_n else 0.0
    result = {"acc": acc, "n": int(total_n)}
    
    if output_dir:
        from pathlib import Path
        import json
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Save test results
        with open(outdir / "test_results.json", "w") as f:
            json.dump({
                "test_acc": acc,
                "test_samples": int(total_n),
                "weights_path": weights_path,
                "model_name": model_name
            }, f, indent=2)
        
        result["saved_to"] = str(outdir / "test_results.json")
    
    return result

