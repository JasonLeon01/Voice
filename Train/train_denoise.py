import os
import random
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from denoise_dataset import DenoiseDataset
from denoise_model import DenoiseAttentionModel

def chunk_tensor(tensor, chunk_size):
    # tensor: [batch, channel, time]
    batch, channel, time = tensor.shape
    num_chunks = (time + chunk_size - 1) // chunk_size
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, time)
        chunk = tensor[:, :, start:end]
        # 如果最后一个chunk不足chunk_size，pad到chunk_size
        if chunk.shape[2] < chunk_size:
            pad_size = chunk_size - chunk.shape[2]
            chunk = torch.nn.functional.pad(chunk, (0, pad_size))
        chunks.append(chunk)
    return chunks

def collate_fn(batch, chunk_size=16000):
    noisy_list, clean_list = zip(*batch)
    lengths = [x.shape[1] for x in noisy_list]
    noisy_padded = pad_sequence([x.t() for x in noisy_list], batch_first=True).transpose(1, 2)
    clean_padded = pad_sequence([x.t() for x in clean_list], batch_first=True).transpose(1, 2)
    # 创建掩码：True 表示填充值
    mask = torch.zeros(noisy_padded.shape[0], noisy_padded.shape[2], dtype=torch.bool)
    for i, length in enumerate(lengths):
        if length < noisy_padded.shape[2]:
            mask[i, length:] = True
    # 分块
    noisy_chunks = chunk_tensor(noisy_padded, chunk_size)
    clean_chunks = chunk_tensor(clean_padded, chunk_size)
    mask_chunks = chunk_tensor(mask.unsqueeze(1).float(), chunk_size)
    mask_chunks = [chunk.squeeze(1).bool() for chunk in mask_chunks]
    return noisy_chunks, clean_chunks, mask_chunks

def main(max_samples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = DenoiseDataset('../Dataset/Output')

    total_len = len(dataset)
    val_num = min(1000, total_len)

    # 前1000组为验证集，其余为训练集
    val_dataset = Subset(dataset, range(val_num))
    train_indices = list(range(val_num, total_len))
    if max_samples is not None and max_samples < len(train_indices):
        train_indices = random.sample(train_indices, max_samples)
    train_dataset = Subset(dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4)

    sample_noisy, _ = dataset[0]
    model = DenoiseAttentionModel(input_dim=sample_noisy.shape[0]).to(device)
    if torch.cuda.is_available():
        original_device = torch.cuda.current_device()
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.init()
        torch.cuda.set_device(original_device)
        print(f'Reset device to {original_device}')

        if torch.cuda.device_count() > 1:
            print(f'{torch.cuda.device_count()} GPUs DataParallel training.')
            model = torch.nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)#1e-3)
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.L1Loss()

    num_epochs = 20
    num_batches = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        chunk_total = 0
        batch_idx = 0
        for noisy_chunks, clean_chunks, mask_chunks in train_loader:
            batch_idx += 1
            for noisy, clean, mask in zip(noisy_chunks, clean_chunks, mask_chunks):
                noisy = noisy.to(device)
                clean = clean.to(device)
                mask = mask.to(device)
                output = model(noisy, src_key_padding_mask=mask)
                print("train output:", output.max().item(), output.min().item())
                loss = loss_fn(output, clean)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                chunk_total += 1
            # print(f"\rEpoch {epoch+1} Progress: {batch_idx/num_batches*100:.2f}%", end="")
            # print()
        avg_loss = total_loss / chunk_total if chunk_total > 0 else 0
        # print()
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.6f}")
        print()

        torch.save(model.state_dict(), f'./denoise_model_epoch{epoch+1}.pth')

        # 验证
        model.eval()
        val_loss = 0
        val_chunks = 0
        with torch.no_grad():
            for noisy_chunks, clean_chunks, mask_chunks in val_loader:
                for noisy, clean, mask in zip(noisy_chunks, clean_chunks, mask_chunks):
                    noisy = noisy.to(device)
                    clean = clean.to(device)
                    mask = mask.to(device)
                    output = model(noisy, src_key_padding_mask=mask)
                    valid_mask = ~mask.unsqueeze(1)
                    output_valid = output.masked_select(valid_mask)
                    clean_valid = clean.masked_select(valid_mask)
                    if output_valid.numel() == 0:
                        continue
                    loss = loss_fn(output_valid, clean_valid)
                    val_loss += loss.item()
                    val_chunks += 1
        avg_val_loss = val_loss / val_chunks if val_chunks > 0 else 0
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.6f}")

if __name__ == "__main__":
    main(1000)