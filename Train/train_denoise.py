import torch
from torch.utils.data import DataLoader
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

def main(max_samples=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = DenoiseDataset('./Output')
    if max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), max_samples)))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    sample_noisy, _ = dataset[0]
    model = DenoiseAttentionModel(input_dim=sample_noisy.shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        chunk_total = 0  # 新增：统计chunk数量
        for noisy_chunks, clean_chunks in dataloader:
            for noisy, clean in zip(noisy_chunks, clean_chunks):
                noisy = noisy.to(device)
                clean = clean.to(device)
                output = model(noisy)
                loss = loss_fn(output, clean)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                chunk_total += 1  # 每处理一个chunk加1
        avg_loss = total_loss / chunk_total  # 用chunk总数做平均
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
        torch.save(model.state_dict(), f'd:\\Dataset\\denoise_model_epoch{epoch+1}.pth')

if __name__ == "__main__":
    main()