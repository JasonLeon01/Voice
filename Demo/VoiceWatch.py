import io
import threading
import time
from librosa import segment
import numpy as np
import sounddevice as sd
import torch
from LLM import LLM
import noisereduce as nr
import whisper
import webrtcvad
from Resemblyzer.resemblyzer import VoiceEncoder, preprocess_wav
import speech_recognition as sr
import soundfile as sf

class Listener:
    def __init__(self, GUI_update_callback):
        self.device = f'cuda' if torch.cuda.is_available() else 'cpu'
        self.GUI_update_callback = GUI_update_callback

        self.language_prompt = {
            "zh": "以下是机场值机柜台的对话内容。用户可能会说'你好小助手'来开始对话。内容主要涉及：航线（例如：北京飞上海、重庆到广州）、航班号（例如：MU2501）、座位类型（头等舱、商务舱、经济舱）、座位号（例如：15A、22C）、座位偏好（靠窗、靠走廊/过道）、登机口号码、航班时间等信息。请准确识别地名、数字和字母，避免同音字混淆。如果没有识别到任何有效信息，则不要输出。",
            "en": "The following is a conversation at an airport check-in counter. The user may say 'Hi Assistant' to start the conversation. Content mainly includes: flight routes (e.g., Beijing to Shanghai, Chongqing to Guangzhou), flight numbers (e.g., MU2501), seat classes (First Class, Business Class, Economy Class), seat numbers (e.g., 15A, 22C), seat preferences (window seat, aisle seat), gate numbers, and flight times. Please accurately recognize place names, numbers and letters. If no valid information is recognized, do not output."
        }

        self.thread = None

        # 录音参数
        self.SAMPLERATE = 16000  # 采样率
        self.CHANNELS = 1  # 单声道

        self.whisper_model = whisper.load_model("turbo")
        self.recognizer = sr.Recognizer()

        self.target_embedding = None
        self.record_embedding = False
        self.last_detected_time = None
        self.speech_detected = False
        self.voiceprint = Voiceprint()
        self.detected_audio = None

        self.frame_duration_ms = 30  # 每帧持续30毫秒
        self.frame_length = int(self.SAMPLERATE * self.frame_duration_ms / 1000)  # 每帧的采样点数
        self.buffer = []  # 用来存储累积的音频数据，但只处理每帧的数据
        self.vad = webrtcvad.Vad(3)

    def denoise_audio(self, audio_data):
        noise = audio_data[: min(len(audio_data), int(0.5 * self.SAMPLERATE))]
        reduced_noise = nr.reduce_noise(y=audio_data, sr=self.SAMPLERATE, y_noise=noise)
        return reduced_noise

    def detect_speech(self):
        if len(self.buffer) < self.frame_length:  # 如果积累的数据不够一帧的大小
            return False, None

        # 获取一帧数据（每帧30ms的数据）
        frame = self.buffer[:self.frame_length]
        self.buffer = self.buffer[self.frame_length:]  # 移除已处理的帧数据

        # 检测这帧数据是否含有语音
        frame_int16 = np.int16(np.array(frame) * 32767).flatten()
        is_speech = self.vad.is_speech(frame_int16.tobytes(), self.SAMPLERATE)
        return is_speech, frame

    def is_speech(self, audio_array):
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        # 设置参数
        frame_duration_ms = 30  # 改为30ms per frame
        frame_length = int(self.SAMPLERATE * frame_duration_ms / 1000)

        # 补足音频长度为frame_length的整数倍
        remainder = len(audio_array) % frame_length
        if remainder != 0:
            padding_length = frame_length - remainder
            audio_array = np.pad(audio_array, (0, padding_length), mode='constant')

        # 切分音频
        num_frames = len(audio_array) // frame_length
        speech_frames = []

        # 处理每个30ms的片段
        for i in range(num_frames):
            frame = audio_array[i * frame_length:(i + 1) * frame_length]
            frame_int16 = (frame * 32767).astype(np.int16)

            try:
                # 判断是否有人声
                if self.vad.is_speech(frame_int16.tobytes(), self.SAMPLERATE):
                    speech_frames.append(frame)
            except Exception as e:
                print(f"Frame {i} processing error: {e}")
                continue

        if not speech_frames:
            return False, None

        # 合并有人声的片段
        speech_audio = np.concatenate(speech_frames)

        # 判断有人声部分是否超过1秒
        if len(speech_audio) > self.SAMPLERATE:  # 超过1秒
            return True, speech_audio
        else:
            return False, None

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        # 将音频数据转换为16-bit PCM数据并累积

        self.buffer.extend(indata.flatten())  # 将数据追加到 buffer

    def listen(self):
        if self.thread is None:
            return
        with sr.Microphone() as source:
            print("Start watching...")
            try:
                audio = self.recognizer.listen(source, timeout=2)
                if self.thread is None:
                    return
            except sr.WaitTimeoutError:
                print("Timeout waiting for audio.")
                if self.thread is None:
                    self.stop_listening()
                    return
                self.listen()
                return

        print("Watching ended.")
        if self.thread is None:
            return
        wav_bytes = audio.get_wav_data(convert_rate=self.SAMPLERATE)
        wav_stream = io.BytesIO(wav_bytes)
        audio_array, sampling_rate = sf.read(wav_stream)
        audio_array = audio_array.astype(np.float32)

        try:
            if not self.record_embedding:
                current_embedding = self.voiceprint.extract_voice_embedding(audio_array)
                self.target_embedding = current_embedding
            else:
                segment_duration = 0.5
                segments = self.voiceprint.split_audio(audio_array, segment_duration)
                similar_segments = []
                for i, segment in enumerate(segments):
                    segment_embedding = self.voiceprint.extract_voice_embedding(segment)
                    similarity = self.voiceprint.get_similarity(segment_embedding, self.target_embedding)
                    print(f"Similarity for segment {i}: {similarity}")
                    if similarity > 0.6:
                        similar_segments.append(segment)
                if similar_segments:
                    print("Similar segments detected.")
                    audio_array = np.concatenate(similar_segments)
                    print(f"Found {len(similar_segments)} segments.")
                else:
                    print("No similar segments found.")
                    self.GUI_update_callback("")
                    self.stop_listening()
                    return

        except Exception as e:
            print(f"Error during voiceprint extraction: {e}")
            self.GUI_update_callback("")
            self.stop_listening()
            return

        has_speech, speech_audio = self.is_speech(audio_array)
        if has_speech:
            print("Speech detected.")
            text, language = self.language_detect(speech_audio)
            if LLM.check_if_gibberish(text) == True:
                print("It's gibberish.")
                self.GUI_update_callback("")
            else:
                self.GUI_update_callback(LLM.auto_fix_dialogue(text))
        else:
            print("It's not person speaking.")
            self.GUI_update_callback("")
        self.stop_listening()

    def language_detect(self, audio_data):
        if self.device == 'cuda':
            fp16 = True
        else:
            fp16 = False

        audio_pot = whisper.pad_or_trim(audio_data)
        mel = whisper.log_mel_spectrogram(audio_pot, n_mels=self.whisper_model.dims.n_mels).to(self.whisper_model.device)
        _, probs = self.whisper_model.detect_language(mel)

        language = max(probs, key=probs.get)
        print(f"Detected language: {language}")

        if language in self.language_prompt:
            initial_prompt = self.language_prompt[language]
        else:
            initial_prompt = "Show the content in the detected language."

        result = self.whisper_model.transcribe(
            audio_data,
            language=language,
            fp16=fp16,
            temperature=0.2,
            initial_prompt=initial_prompt
        )

        print(f"Transcription result: {result['text']}")
        return result["text"], language

    def start_listening(self):
        self.thread = threading.Thread(target=self.listen)
        self.thread.start()

    def stop_listening(self):
        sd.stop()
        self.thread = None
        print("Stop watching...")

class Voiceprint:
    def __init__(self):
        self.encoder = VoiceEncoder()
        self.SAMPLERATE = 16000

    def extract_voice_embedding(self, audio_data):
        print("Extracting voice embedding...")
        wav = preprocess_wav(audio_data, self.SAMPLERATE)  # 预处理音频数据
        embedding = self.encoder.embed_utterance(wav)  # 提取声纹特征
        print("Voice embedding extracted.")
        return embedding

    def split_audio(self, audio_data, segment_duration):
        segment_length = int(self.SAMPLERATE * segment_duration)
        return [audio_data[i:i + segment_length] for i in range(0, len(audio_data), segment_length)]

    def get_similarity(self, embedding1, embedding2):
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return similarity
