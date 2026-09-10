from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_curve
from torch.utils.tensorboard import SummaryWriter
from mcd_model import MCDC
from utils import *


# ---------------------------------------------------------------------------
# Training / validation epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, is_train,
              class_weights=None, scaler=None):
    """Single epoch. Returns (avg_loss, eer, accuracy)."""
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n          = 0
    correct    = 0
    all_scores = []
    all_labels = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for x, y in tqdm(loader, desc="train" if is_train else "val ", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # ---- forward pass with AMP ----
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(scaler is not None)):
                logits = model(x)
                loss   = F.cross_entropy(logits, y, weight=class_weights)

            # ---- backward pass ----
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)             # unscale before clipping
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

            total_loss += loss.item() * x.size(0)
            n          += x.size(0)
            correct    += (logits.argmax(dim=1) == y).sum().item()

            prob_real = torch.softmax(logits.float(), dim=1)[:, 0].detach().cpu().numpy()
            target    = (y == 0).long().cpu().numpy()
            all_scores.append(prob_real)
            all_labels.append(target)

    avg_loss   = total_loss / n
    accuracy   = correct / n
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    eer        = compute_eer(all_scores, all_labels)

    return avg_loss, eer, accuracy


# ---------------------------------------------------------------------------
# MC Dropout validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_mc_eval(model, loader, device, n_samples=30):
    """
    Full MC Dropout pass over a loader.
    Returns EER from mean MC probabilities + mean BALD uncertainty.
    """
    model.train()   # dropout ON

    all_scores = []
    all_labels = []
    all_bald   = []

    for x, y in tqdm(loader, desc="mc  ", leave=False):
        x = x.to(device, non_blocking=True)
        mean_prob, _, bald = model.mc_predict(x, n_samples=n_samples)
        all_scores.append(mean_prob[:, 0].cpu().numpy())
        all_labels.append((y == 0).long().numpy())
        all_bald.append(bald.cpu().numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    all_bald   = np.concatenate(all_bald)

    return compute_eer(all_scores, all_labels), float(all_bald.mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    train_manifest : str,
    batch_size     : int   = 64,
    lr             : float = 3e-4,
    epochs         : int   = 30,
    num_workers    : int   = 4,
    mc_samples     : int   = 30,
    mc_eval_every  : int   = 5,
    ckpt_path      : str   = "mcdc.pt",
    log_dir        : str   = "runs/mcdc",
    use_amp        : bool  = True,    # set False if not on CUDA
):
    # ---- device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False   # AMP not supported on MPS
    else:
        device = torch.device("cpu")
        use_amp = False   # AMP not useful on CPU
    print(f"Device : {device}  |  AMP : {use_amp}")

    # ---- AMP scaler (only used when use_amp=True) ----
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard  → {log_dir}  |  tensorboard --logdir={log_dir}")

    # ---- dataset — balanced subset ----
    df       = pd.read_csv(train_manifest)
    
    n_real = int((df["label"] == 0).sum())
    n_fake = int((df["label"] == 1).sum())
    print(f"Train  — real: {n_real}  fake: {n_fake}  ratio 1:{n_fake // n_real}")
    
    train_ds     = SpecDataset(train_manifest)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_trim_whisper, num_workers=num_workers,
        pin_memory=True,
    )
    
    class_weights = torch.tensor([n_fake / n_real, 1.0]).to(device)

    val_ds = AudioDataset(
        protocol_file = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        flac_root     = "LA/ASVspoof2019_LA_dev/flac",
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_trim_whisper, num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Validation - {len(val_ds)} samples")
    
    # ---- model ----
    model   = MCDC(p_drop=0.3).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {n_param / 1e6:.2f}M")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    #model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    # ---- optimiser + scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for _ in range(29):
        scheduler.step()

    best_val_eer = 1.0
    epoch = 30

    # ---- training loop ----
    for epoch in range(epoch + 1, epochs + 1):
        train_loss, train_eer, train_acc = run_epoch(
            model, train_loader, optimizer, device,
            is_train=True, class_weights=class_weights, scaler=scaler,
        )
        scheduler.step()

        # -- TensorBoard: train metrics --
        writer.add_scalar("Loss/train",     train_loss,      epoch)
        writer.add_scalar("EER/train",      train_eer * 100, epoch)
        writer.add_scalar("Accuracy/train", train_acc * 100, epoch)
        writer.add_scalar("LR",             scheduler.get_last_lr()[0], epoch)


        log = (
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss {train_loss:.4f} | "
            f"EER {train_eer * 100:.2f}% | "
            f"Acc {train_acc * 100:.2f}%"
        )

        # ---- validation ----
        if epoch % 3 == 0:
            val_loss, val_eer, val_acc = run_epoch(
                model, val_loader, optimizer, device, is_train=False,
            )

            writer.add_scalar("Loss/val",     val_loss,      epoch)
            writer.add_scalar("EER/val",      val_eer * 100, epoch)
            writer.add_scalar("Accuracy/val", val_acc * 100, epoch)
            writer.add_scalars("EER/train_vs_val", {
                "train": train_eer * 100,
                "val":   val_eer   * 100,
            }, epoch)
            writer.add_scalars("Loss/train_vs_val", {
                "train": train_loss,
                "val":   val_loss,
            }, epoch)

            log += (
                f" | val loss {val_loss:.4f} | "
                f"val EER {val_eer * 100:.2f}% | "
                f"val Acc {val_acc * 100:.2f}%"
            )
            torch.save(model.state_dict(), f"mcdc_epoch{epoch:02d}.pt")
            print("Weights saved")

            # ---- MC Dropout eval every mc_eval_every epochs ----
            """
            if epoch % mc_eval_every == 0:
                mc_eer, mean_bald = run_mc_eval(
                    model, val_loader, device, n_samples=mc_samples,
                )
                writer.add_scalar("EER/mc_val",    mc_eer    * 100, epoch)
                writer.add_scalar("BALD/mean_val", mean_bald,       epoch)
                writer.add_scalars("EER/mc_vs_standard", {
                    "mc":       mc_eer  * 100,
                    "standard": val_eer * 100,
                }, epoch)
                log += f" | MC EER {mc_eer * 100:.2f}% | BALD {mean_bald:.4f}"
            
            
            # ---- checkpoint on best val EER ----
            if val_eer < best_val_eer:
                best_val_eer = val_eer
                torch.save(model.state_dict(), ckpt_path)
                writer.add_scalar("EER/best_val", best_val_eer * 100, epoch)
                log += "  ← saved"
            """
        print(log)

    writer.close()
    print(f"\nBest val EER : {best_val_eer * 100:.2f}%")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"TensorBoard  : tensorboard --logdir={log_dir}")


if __name__ == "__main__":
    main(
        train_manifest = "LA/features/train/manifest.csv",
        batch_size     = 64,
        lr             = 3e-4,
        epochs         = 40,
        num_workers    = 2,
        mc_samples     = 30,
        mc_eval_every  = 5,
        ckpt_path      = "mcdc_epoch30.pt",
        log_dir        = "runs/mcdc",
        use_amp        = True,
    )