import os
import torch
from torch.utils.data import Dataset
import torchaudio

class DenoiseDataset(Dataset):
    def __init__(self, root_dir):
        self.pairs = []
        for group in os.listdir(root_dir):
            group_path = os.path.join(root_dir, group)
            origin_path = os.path.join(group_path, "origin.wav")
            result_path = os.path.join(group_path, "result.wav")
            if os.path.exists(origin_path) and os.path.exists(result_path):
                self.pairs.append((result_path, origin_path))
        self.length = len(self.pairs)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy_waveform, _ = torchaudio.load(noisy_path)
        clean_waveform, _ = torchaudio.load(clean_path)
        min_len = min(noisy_waveform.shape[1], clean_waveform.shape[1])
        noisy_waveform = noisy_waveform[:, :min_len]
        clean_waveform = clean_waveform[:, :min_len]
        # 如果clean是单通道，扩展为双通道
        if clean_waveform.shape[0] == 1 and noisy_waveform.shape[0] == 2:
            clean_waveform = clean_waveform.repeat(2, 1)
        return noisy_waveform, clean_waveform