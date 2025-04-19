import torch
import torchaudio
import os
import random
import shutil
from denoise_model import DenoiseAttentionModel

def denoise_one(noisy_path, model, device):
    noisy_waveform, sr = torchaudio.load(noisy_path)
    input_dim = noisy_waveform.shape[0]
    noisy_waveform = noisy_waveform.unsqueeze(0).to(device)  # [1, channels, time]
    with torch.no_grad():
        denoised = model(noisy_waveform)
    denoised = denoised.squeeze(0).cpu()
    return denoised, sr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = '../Dataset/Output'
    # 获取前1000组文件夹
    all_folders = sorted(os.listdir(output_dir))
    val_folders = all_folders[:1000] if len(all_folders) >= 1000 else all_folders
    if not val_folders:
        raise ValueError("No folders found in the output directory.")
    # 随机选一组
    folder = random.choice(val_folders)
    group_path = os.path.join(output_dir, folder)
    result_path = os.path.join(group_path, "result.wav")
    if not os.path.exists(result_path):
        print(f"{result_path} not found.")
        return

    # 获取通道数并加载模型
    sample_waveform, _ = torchaudio.load(result_path)
    input_dim = sample_waveform.shape[0]
    model = DenoiseAttentionModel(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load('./denoise_model_epoch20.pth', map_location=device))
    model.eval()

    # 去噪并直接保存到根目录
    denoised, sr = denoise_one(result_path, model, device)
    denoised_path = "./denoised.wav"
    torchaudio.save(denoised_path, denoised, sr)

    # 复制result.wav到根目录
    result_copy_path = "./result.wav"
    shutil.copyfile(result_path, result_copy_path)

    print(f"Processed: {result_path}")
    print(f"Saved denoised audio to: {denoised_path}")
    print(f"Copied original result.wav to: {result_copy_path}")

if __name__ == "__main__":
    main()