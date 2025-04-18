# separation_model = TDANet.look2hear.models.BaseModel.from_pretrain(
#     "./TDANet/local_models/TDANetBest-4ms-LRS2/pytorch_model.bin"
# ).to(device)
# separation_model.eval()

# 分离音频的函数
# def separate_audio(mix_audio):
#     print("Start separating...")
#     mix_audio = torch.from_numpy(mix_audio).view(1, 1, -1)  # 转换为 PyTorch tensor
#     with torch.no_grad():  # 禁用梯度计算
#         est_sources = separation_model(mix_audio.to(device))
#     print("Separation finished.")
#     return est_sources

# 并行处理识别任务
# def parallel_recognize(audio_data_1, audio_data_2, sample_rate, result_1, result_2):
#     with ThreadPoolExecutor(max_workers=2) as executor:
#         # 提交任务，并将结果存入各自的容器
#         executor.submit(recognize_audio, audio_data_1, sample_rate, result_1)
#         executor.submit(recognize_audio, audio_data_2, sample_rate, result_2)

    # # 分离音频
    # est_sources = separate_audio(audio_data)

    # # 转为 NumPy 数组
    # audio_data_1 = (est_sources[:, 0, :].detach().cpu() * 32767).numpy().astype(np.int16)
    # audio_data_2 = (est_sources[:, 1, :].detach().cpu() * 32767).numpy().astype(np.int16)

    # # 并行识别
    # result_1, result_2 = [], []
    # parallel_recognize(audio_data_1, audio_data_2, SAMPLERATE, result_1, result_2)

    # # 输出结果
    # print("Voice 1:", result_1[0])
    # print("Voice 2:", result_2[0])
