import importlib, json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def get_device():
    """Auto-detect best available device across all systems."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"



def train_model(
        *,
        model_name: str,
        model_kwargs: Dict,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 20,
        lr: float = 1e-3,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        stopping_criteria: bool = False
) -> Dict:
    mod = importlib.import_module(f"src.models.{model_name}")
    ModelClass = getattr(mod, "Model")
    model: nn.Module = ModelClass(**model_kwargs)

    if device is None:
        device = get_device()
    print(f"Using device: {device}")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    patience = 5
    min_impr = 1e-3
    no_improve = 0

    outdir = Path(output_dir) if output_dir else None
    if outdir:
        (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
        with open(outdir / "config.json", "w") as f:
            json.dump({
                "model_name": model_name,
                "model_kwargs": model_kwargs,
                "num_epochs": num_epochs,
                "lr": lr,
                "device": device,
                "early_stopping": {"enabled": bool(stopping_criteria), "patience": patience, "min_impr": min_impr},
                "selection": "best_val_acc"
            }, f, indent=2)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for batch in train_loader:
            if len(batch) < 2:
                raise ValueError("Each batch must be at least (x,y)")
            x, y = batch[0].to(device), batch[1].to(device)
            if x.ndim == 2:
                x = x.unsqueeze(1)
            elif x.ndim != 3:
                raise ValueError(f"Expected x with 2 or 3 dims, got shape {tuple(x.shape)}")

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * y.numel()
            total_correct += (preds == y).sum().item()
            total_n += y.numel()

        train_loss = total_loss / total_n
        train_acc = total_correct / total_n
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                v_loss, v_correct, v_n = 0.0, 0, 0
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    if xb.ndim == 2:
                        xb = xb.unsqueeze(1)
                    logits = model(xb)
                    v_loss += criterion(logits, yb).item() * yb.numel()
                    v_correct += (logits.argmax(dim=1) == yb).sum().item()
                    v_n += yb.numel()

                total_val_loss = v_loss / v_n
                total_val_acc = v_correct / v_n
                history["val_loss"].append(total_val_loss)
                history["val_acc"].append(total_val_acc)

                if total_val_acc > best_val_acc + min_impr:
                    best_val_acc = total_val_acc
                    no_improve = 0
                    if outdir:
                        torch.save(model.state_dict(), (outdir / "checkpoints" / "best.pt").as_posix())
                else:
                    if stopping_criteria:
                        no_improve += 1

            print(f"[{epoch:03d}] train loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val loss={total_val_loss:.4f} acc={total_val_acc:.3f}")

            if stopping_criteria and no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val acc: {best_val_acc:.4f})")
                break
        else:
            print(f"[{epoch:03d}] train loss={train_loss:.4f} acc={train_acc:.3f}")

    last_ckpt = None
    best_ckpt = None
    metrics_path = None
    if outdir:
        last_path = outdir / "checkpoints" / "last.pt"
        torch.save(model.state_dict(), last_path.as_posix())
        last_ckpt = last_path.as_posix()
        bp = outdir / "checkpoints" / "best.pt"
        if bp.exists():
            best_ckpt = bp.as_posix()
        metrics_path = (outdir / "metrics.json").as_posix()
        with open(metrics_path, "w") as f:
            json.dump({"history": history, "best_val_acc": best_val_acc}, f, indent=2)

    return {
        "history": history,
        "best_val_acc": best_val_acc if best_val_acc >= 0 else None,
        "best_ckpt": best_ckpt,
        "last_ckpt": last_ckpt,
        "paths": {"metrics_json": metrics_path, "config_json": (outdir / "config.json").as_posix() if outdir else None},
    }
