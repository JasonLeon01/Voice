import os
import random
import shutil
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_all_files(folder, exts=None):
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if exts is None or f.lower().endswith(tuple(exts)):
                files.append(os.path.join(root, f))
    return files

def process_one(args):
    clean_file, noise_segments, group_idx, output_dir = args
    group_output = os.path.join(output_dir, f"output_{group_idx}")
    os.makedirs(group_output, exist_ok=True)
    clean_audio = AudioSegment.from_file(clean_file)
    clean_len = len(clean_audio)
    noise_audio = random.choice(noise_segments)
    if len(noise_audio) <= clean_len:
        start = 0
    else:
        start = random.randint(0, len(noise_audio) - clean_len)
    noise_clip = noise_audio[start:start + clean_len] - 6
    mixed = clean_audio.overlay(noise_clip)
    result_path = os.path.join(group_output, "result.wav")
    mixed.export(result_path, format="wav")
    origin_path = os.path.join(group_output, "origin.wav")
    shutil.copy2(clean_file, origin_path)
    return clean_file, group_output

def main():
    noise_dir = os.path.join(os.getcwd(), "Noise")
    total_dir = os.path.join(os.getcwd(), "TotalFolder")
    output_dir = os.path.join(os.getcwd(), "Output")
    if os.path.exists(output_dir):
        print("Output folder already exists. Deleting...")
        shutil.rmtree(output_dir)
        print("Output folder deleted.")
    os.makedirs(output_dir, exist_ok=True)

    noise_files = get_all_files(noise_dir, exts=[".wav", ".mp3"])
    noise_segments = [AudioSegment.from_file(f) for f in noise_files]

    group_folders = [os.path.join(total_dir, d) for d in os.listdir(total_dir) if os.path.isdir(os.path.join(total_dir, d))]

    tasks = []
    group_idx = 0
    for group in group_folders:
        clean_files = get_all_files(group, exts=[".wav", ".mp3"])
        for clean_file in clean_files:
            group_idx += 1
            tasks.append((clean_file, noise_segments, group_idx, output_dir))
    tasks = tasks[:100000]

    total = len(tasks)
    print(f"All tasks: {total}")

    finished = 0
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_one, task) for task in tasks]
        for future in as_completed(futures):
            clean_file, group_output = future.result()
            finished += 1
            print(f"[{finished}/{total}] Processed: {clean_file} -> {group_output}", flush=True)

if __name__ == "__main__":
    main()