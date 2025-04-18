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
    noisy_padded = pad_sequence([x.t() for x in noisy_list], batch_first=True).transpose(1,2)
    clean_padded = pad_sequence([x.t() for x in clean_list], batch_first=True).transpose(1,2)
    # 分块
    noisy_chunks = chunk_tensor(noisy_padded, chunk_size)
    clean_chunks = chunk_tensor(clean_padded, chunk_size)
    # 每个chunk shape: [batch, channel, chunk_size]
    return noisy_chunks, clean_chunks

def main(max_samples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = DenoiseDataset('./Output')
    total_len = len(dataset)
    val_num = min(1000, total_len)
    train_num = total_len - val_num

    # 前1000组为验证集，其余为训练集
    val_dataset = Subset(dataset, range(val_num))
    train_indices = list(range(val_num, total_len))
    if max_samples is not None and max_samples < len(train_indices):
        train_indices = random.sample(train_indices, max_samples)
    train_dataset = Subset(dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    sample_noisy, _ = dataset[0]
    model = DenoiseAttentionModel(input_dim=sample_noisy.shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        chunk_total = 0
        for noisy_chunks, clean_chunks in train_loader:
            for noisy, clean in zip(noisy_chunks, clean_chunks):
                noisy = noisy.to(device)
                clean = clean.to(device)
                output = model(noisy)
                loss = loss_fn(output, clean)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                chunk_total += 1
        avg_loss = total_loss / chunk_total if chunk_total > 0 else 0
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.6f}")

        # 验证
        model.eval()
        val_loss = 0
        val_chunks = 0
        with torch.no_grad():
            for noisy_chunks, clean_chunks in val_loader:
                for noisy, clean in zip(noisy_chunks, clean_chunks):
                    noisy = noisy.to(device)
                    clean = clean.to(device)
                    output = model(noisy)
                    loss = loss_fn(output, clean)
                    val_loss += loss.item()
                    val_chunks += 1
        avg_val_loss = val_loss / val_chunks if val_chunks > 0 else 0
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.6f}")

        torch.save(model.state_dict(), f'./denoise_model_epoch{epoch+1}.pth')

if __name__ == "__main__":
    main(10000)