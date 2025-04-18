import os
import torchaudio

output_dir = './Output'
single_channel_files = []
double_channel_files = []
other_channel_files = []

for folder in os.listdir(output_dir):
    group_path = os.path.join(output_dir, folder)
    for wav_name in ["origin.wav", "result.wav"]:
        wav_path = os.path.join(group_path, wav_name)
        if os.path.exists(wav_path):
            try:
                waveform, _ = torchaudio.load(wav_path)
                channels = waveform.shape[0]
                if channels == 1:
                    single_channel_files.append(wav_path)
                elif channels == 2:
                    double_channel_files.append(wav_path)
                else:
                    other_channel_files.append((wav_path, channels))
            except Exception as e:
                print(f"Error loading {wav_path}: {e}")

print("1 channel:")
for f in single_channel_files:
    print(f)

print("\n2 channel:")
for f in double_channel_files:
    print(f)

if other_channel_files:
    print("\nother channel:")
    for f, ch in other_channel_files:
        print(f"{f} (channels={ch})")