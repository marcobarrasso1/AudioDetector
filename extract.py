import torch
import torchaudio
from transformers import WhisperFeatureExtractor
import os
import csv
from tqdm import tqdm
import soundfile as sf
import numpy as np

fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

def extract_whisper_features(wav: torch.Tensor, sr: int = 16000):
    # wav: [C,T] or [T]; returns (spec, mask)
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    if sr != fe.sampling_rate:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=fe.sampling_rate)
    wav_np = wav.cpu().numpy()
    inputs = fe(
        wav_np,
        sampling_rate=fe.sampling_rate,
        return_tensors="pt",
        return_attention_mask=True,      # <-- important
    )
    spec = inputs.input_features[0]      # [1,80,3000]
    mask = inputs.attention_mask[0]      # [3000], 1=real, 0=pad (dtype: long/bool)
    return spec, mask


def safe_load_audio_sf(path: str):
    """
    Load audio with soundfile and return a torch.Tensor [C, T], sr.
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # data: [T, C]
    data = np.transpose(data)  # [C, T]
    wav = torch.from_numpy(data)  # float32
    return wav, sr


def pre_extract_split(flac_root: str, protocol_file: str, output_dir: str):
    """
    Pre-extract Whisper features for a given split and save to disk with a manifest.

    Args:
        wav_root: Directory containing WAV files named <utterance_id>.wav
        protocol_file: Path to the protocol .txt file (e.g., .trl.txt or .trn.txt) listing fields
        output_dir: Directory where .pt feature files and manifest.csv will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_rows = []

    # Read all protocol lines so tqdm can show total
    with open(protocol_file, 'r') as f:
        lines = f.readlines()
    '''
    for line in tqdm(lines, desc=f"Extracting features to {output_dir}", unit="file"):
            parts = line.strip().split()
            # parts example: [speaker_id, utterance_id, -, -, label]
            utt_id = parts[1]
            label_str = parts[-1]
            label = 0 if label_str == 'bonafide' else 1
            audio_path = os.path.join(flac_root, f"{utt_id}.flac")
            flac, sr = torchaudio.load(audio_path)
            spec = extract_whisper_features(flac, sr)  # [1, 80, T]
            out_file = os.path.join(output_dir, f"{utt_id}.pt")
            torch.save(spec, out_file)
            manifest_rows.append([out_file, label])
    '''
    
    for line in tqdm(lines, desc=f"Extracting features to {output_dir}", unit="file"):
        parts = line.strip().split()
        utt_id = parts[1]
        label_str = parts[-1]
        label = 0 if label_str == 'bonafide' else 1

        audio_path = os.path.join(flac_root, f"{utt_id}.flac")
        wav, sr = safe_load_audio_sf(audio_path)
        spec, mask = extract_whisper_features(wav, sr)
        out_file = os.path.join(output_dir, f"{utt_id}.pt")
        torch.save({"spec": spec, "mask": mask}, out_file)  # <-- save both
        manifest_rows.append([out_file, label])
    
    # Write manifest.csv
    manifest_path = os.path.join(output_dir, 'manifest.csv')
    with open(manifest_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filepath', 'label'])
        writer.writerows(manifest_rows)

    print(f"Pre-extracted features saved to {output_dir}, manifest at {manifest_path}")

def main():
    
    pre_extract_split(
    flac_root='LA/ASVspoof2019_LA_dev/flac',
    protocol_file='LA/ASVspoof2019.LA.cm.dev.subset.txt',
    output_dir='LA/features/dev'
    )
    
    '''
    pre_extract_split(
    flac_root='LA/ASVspoof2019_LA_eval/flac',
    protocol_file='LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
    output_dir='LA/features/eval'
    )
    '''
if __name__ == "__main__":
    main()
 