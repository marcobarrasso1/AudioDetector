import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve
import csv
import os
from transformers import WhisperFeatureExtractor
import soundfile as sf
from extract import safe_load_audio_sf, extract_whisper_features


class SpecDataset(Dataset):
    """Loads pre-extracted Whisper spectrograms saved as .pt tensors.
       Each row in the manifest CSV must have: filepath,label
       where filepath points to a .pt file containing a tensor [1, 80, T].
    """
    def __init__(self, manifest_csv: str):
        self.samples = []
        with open(manifest_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row['filepath'], int(row['label'])))
        if not self.samples:
            raise RuntimeError(f"Manifest has no rows: {manifest_csv}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        spec_path, label = self.samples[idx]
        x = torch.load(spec_path)   # [1, 80, T]
        return x, torch.tensor(label, dtype=torch.long)

class AudioDataset(Dataset):
    """
    Loads raw .flac audio and computes Whisper spectrograms on-the-fly.
    Use this for eval/test where precomputing would require too much disk space.
    
    Protocol file format: speaker_id utterance_id - attack_type label
    """
    def __init__(self, protocol_file: str, flac_root: str):
        self.samples  = []
        self.flac_root = flac_root

        with open(protocol_file, 'r') as f:
            for line in f:
                parts     = line.strip().split()
                utt_id    = parts[1]
                label     = 0 if parts[-1] == 'bonafide' else 1
                audio_path = os.path.join(flac_root, f"{utt_id}.flac")
                self.samples.append((audio_path, label))

        if not self.samples:
            raise RuntimeError(f"No samples found in {protocol_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        wav, sr           = safe_load_audio_sf(audio_path)
        spec, mask        = extract_whisper_features(wav, sr)
        # return same format as SpecDataset so collate_trim_whisper works unchanged
        return {"spec": spec, "mask": mask}, torch.tensor(label, dtype=torch.long)
    
    

def collate_trim_whisper(batch):
    inputs_list, labels_list = zip(*batch)

    specs = torch.stack([x["spec"] for x in inputs_list], dim=0)   # [B, 80, T]
    masks = torch.stack([x["mask"] for x in inputs_list], dim=0)   # [B, T]

    lengths = masks.sum(dim=-1)
    maxT    = int(lengths.max().item())

    specs  = specs[..., :maxT].unsqueeze(1)     # [B, 1, 80, maxT]
    labels = torch.as_tensor(labels_list)

    return specs, labels


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    scores : prob(real) — higher = more likely real
    labels : 1 = real, 0 = fake  (positive class = real)
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)



