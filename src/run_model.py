# src/run_model.py
import importlib
from typing import Dict, Optional, Union
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

@torch.no_grad()
def run_model(
    *,
    model_name: str,
    model_kwargs: Dict,
    dataloader: DataLoader,
    weights_path: Optional[Union[str, os.PathLike]] = None,  # str or Path
    device,                                                  # required; you pass it in
    output_dir: Optional[Union[str, os.PathLike]] = None,
) -> Dict:
    # Import models same way as train_model does for this tree
    mod = importlib.import_module(f"models.{model_name}")
    ModelClass = getattr(mod, "Model")
    model: nn.Module = ModelClass(**model_kwargs)

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
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total_n += y.numel()

    acc = (correct / total_n) if total_n else 0.0
    result: Dict = {"acc": acc, "n": int(total_n)}

    if output_dir:
        from pathlib import Path
        import json
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / "test_results.json"
        with open(out_path, "w") as f:
            json.dump(
                {
                    "test_acc": acc,
                    "test_samples": int(total_n),
                    "weights_path": str(weights_path) if weights_path is not None else None,
                    "model_name": model_name,
                },
                f,
                indent=2,
            )
        result["saved_to"] = str(out_path)

    return result
