# VoiceRecognition
## Installation
First, install Python 3.10.0 or higher version. Clone this repository and install dependencies and models.

```bash
pip install -r requirements.txt
# If you have GPU
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Otherwise
pip3 install torch torchvision torchaudio
```
## Used Repos
### noisereduce
This repository is used to denoise. It works by computing a spectrogram of a signal (and optionally a noise signal) and estimating a noise threshold (or gate) for each frequency band of that signal/noise.

**It does not rely on machine learning.**

https://github.com/timsainb/noisereduce

### Reseemblyzer
This repository is used to extract voiceprint features and compare the similarity of the voice with the recorded voiceprint.

https://github.com/resemble-ai/Resemblyzer

### OpenAI-Whisper
This repository is used for automatic speech recognition (ASR). It can transcribe and translate audio files into text using state-of-the-art machine learning models.

https://github.com/openai/whisper
