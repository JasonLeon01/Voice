import torch
import torchaudio
import os
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
    # 自动检测通道数
    sample_path = None
    output_dir = './Output'
    for folder in os.listdir(output_dir):
        group_path = os.path.join(output_dir, folder)
        result_path = os.path.join(group_path, "result.wav")
        if os.path.exists(result_path):
            sample_path = result_path
            break
    if sample_path is None:
        print("Not found result.wav")
        return

    # 获取通道数
    sample_waveform, _ = torchaudio.load(sample_path)
    input_dim = sample_waveform.shape[0]
    model = DenoiseAttentionModel(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(r'd:\Dataset\denoise_model_epoch20.pth', map_location=device))
    model.eval()

    # 批量处理所有output_x文件夹
    for folder in os.listdir(output_dir):
        group_path = os.path.join(output_dir, folder)
        result_path = os.path.join(group_path, "result.wav")
        if not os.path.exists(result_path):
            continue
        denoised, sr = denoise_one(result_path, model, device)
        save_path = os.path.join(group_path, "denoised.wav")
        torchaudio.save(save_path, denoised, sr)
        print(f"Processed: {result_path} -> {save_path}")

if __name__ == "__main__":
    main()