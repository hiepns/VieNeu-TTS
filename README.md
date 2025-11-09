# VieNeu-TTS

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/pnnbao97/VieNeu-TTS)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)

[<img width="600" height="595" alt="VieNeu-TTS" src="https://github.com/user-attachments/assets/66c098c4-d184-4e7a-826a-ba8c6c556fab" />](https://github.com/user-attachments/assets/5ad53bc9-e816-41a7-9474-ea470b1cbfdd)

> 📢 **Upcoming Release Announcement**
>
> **VieNeu-TTS-1000h is coming soon!** 🚀  
> We are training an upgraded checkpoint with 1 000 hours of data to deliver:  
> - Better pronunciation accuracy  
> - More natural prosody and intonation  
> - Superior voice-cloning fidelity  
> - Stronger handling of complex Vietnamese text  
>
> **Current release:** VieNeu-TTS-140h (stable & production-ready)  
> Follow this repo or the Hugging Face model card for updates.

**VieNeu-TTS** is an on-device Vietnamese Text-to-Speech system with instant voice cloning. The model is fine-tuned from [NeuTTS Air](https://huggingface.co/neuphonic/neutts-air) and produces natural 24 kHz speech with real-time latency on typical CPUs or GPUs. It is ideal for offline voice assistants, embedded devices, privacy-first applications, and creative tools.

**Author:** Phạm Nguyễn Ngọc Bảo

---

## ✨ Features

- 🎙️ High-quality Vietnamese speech at 24 kHz
- 🚀 Instant voice cloning using a short reference clip
- 💻 Fully offline inference (no internet required)
- 🎯 Multiple curated reference voices (Southern accent, male & female)
- ⚡ Real-time or faster-than-real-time synthesis on CPU/GPU
- 🖥️ Ready-to-use Python API, CLI scripts, and a Gradio UI

---

## 🔬 Model Overview

- **Backbone:** Qwen 0.5B LLM (chat template)
- **Audio codec:** NeuCodec (torch implementation; ONNX & quantized variants supported)
- **Context window:** 2 048 tokens shared by prompt text and speech tokens
- **Output watermark:** Enabled by default
- **Training data:**  
  - [VieNeuCodec-dataset](https://huggingface.co/datasets/pnnbao-ump/VieNeuCodec-dataset) — 74.9 k curated Vietnamese samples  
  - Fine-tuned from a base model trained on [Emilia Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)

---

## 🏁 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/pnnbao97/VieNeu-TTS.git
cd VieNeu-TTS
```

### 2. Install eSpeak NG (required by phonemizer)

Follow the [official installation guide](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md). Common commands:

```bash
# macOS
brew install espeak

# Ubuntu / Debian
sudo apt install espeak-ng

# Arch Linux
paru -S aur/espeak-ng

# Windows
# Download installer from https://github.com/espeak-ng/espeak-ng/releases
# Default path: C:\Program Files\eSpeak NG\
# VieNeu-TTS auto-detects this path.
```

**macOS tips**
- If the phonemizer cannot find the library, set `PHONEMIZER_ESPEAK_LIBRARY` to the `.dylib` path.
- Validate installation with: `echo 'test' | espeak-ng -x -q --ipa -v vi`

### 3. Install Python dependencies (Python ≥ 3.11)

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Optional alternatives
uv pip install -r requirements.txt
pip install -e .
```

If you intend to run on GPU, install the matching CUDA build of PyTorch:

```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📦 Project Structure

```
VieNeu-TTS/
├── examples/
│   ├── infer_long_text.py     # CLI for long-form synthesis (chunked)
│   └── sample_long_text.txt   # Example paragraph for testing
├── gradio_app.py              # Local Gradio demo
├── main.py                    # Basic batch inference script
├── output_audio/              # Generated audio (created when running scripts)
├── sample/                    # Reference voices (audio + transcript pairs)
├── utils/
│   ├── __init__.py
│   └── normalize_text.py      # Vietnamese text normalization pipeline
├── vieneu_tts/
│   ├── __init__.py
│   └── vieneu_tts.py          # Core VieNeuTTS implementation
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## 🚀 Quickstart

### Python API

```python
from pathlib import Path

from utils.normalize_text import VietnameseTTSNormalizer
from vieneu_tts import VieNeuTTS
import soundfile as sf

texts = [
    "Các khóa học trực tuyến đang giúp học sinh tiếp cận kiến thức mọi lúc mọi nơi.",
    "Các nghiên cứu về bệnh Alzheimer cho thấy tác dụng tích cực của các bài tập trí não.",
]

ref_audio_path = "sample/id_0001.wav"
ref_text_path = "sample/id_0001.txt"

normalizer = VietnameseTTSNormalizer()
ref_text_raw = Path(ref_text_path).read_text(encoding="utf-8")
normalized_ref_text = normalizer.normalize(ref_text_raw)

tts = VieNeuTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS",
    backbone_device="cuda",   # or "cpu"
    codec_repo="neuphonic/neucodec",
    codec_device="cuda",
)
ref_codes = tts.encode_reference(ref_audio_path)

output_dir = Path("output_audio")
output_dir.mkdir(exist_ok=True)

for idx, raw_text in enumerate(texts, 1):
    normalized_text = normalizer.normalize(raw_text)
    wav = tts.infer(normalized_text, ref_codes, normalized_ref_text)
    sf.write(output_dir / f"output_{idx}.wav", wav, 24_000)
```

### CLI example (`main.py`)

```bash
python main.py
```

This script runs several normalized sentences using the bundled sample voice and writes `output_*.wav` files under `output_audio/`.

### Gradio web demo

```bash
python gradio_app.py
```

Then open `http://127.0.0.1:7860` to:

- Pick one of six reference voices
- Upload your own reference audio + transcript
- Enter up to 250 characters per request (recommended)
- Preview or download the synthesized audio

### Long-text helper

`examples/infer_long_text.py` chunks long passages into ≤256-character segments (prefers sentence boundaries) and synthesizes them sequentially.

```bash
python -m examples.infer_long_text.py \
  --text-file examples/sample_long_text.txt \
  --ref-audio sample/id_0001.wav \
  --ref-text sample/id_0001.txt \
  --output output_audio/sample_long_text.wav
```

[🎵 Listen to sample (MP3)](https://github.com/user-attachments/files/23436562/longtext.mp3)

Use `--text "raw paragraph here"` to infer without creating a file.

---

## 🔈 Reference Voices (`sample/`)

| File      | Gender | Accent | Description        |
|-----------|--------|--------|--------------------|
| id_0001   | Male   | South  | Male voice 1       |
| id_0002   | Female | South  | Female voice 1     |
| id_0003   | Male   | South  | Male voice 2       |
| id_0004   | Female | South  | Female voice 2     |
| id_0005   | Male   | South  | Male voice 3       |
| id_0007   | Male   | South  | Male voice 4       |

Odd IDs correspond to male voices; even IDs correspond to female voices.

---

## ✅ Best Practices & Limits

- Keep each inference request ≤250 characters to stay within the 2 048-token context window (reference speech tokens also consume context).
- Normalize both the target text and the reference transcript before inference (built-in scripts already do this).
- Trim reference audio to ~3–5 seconds for faster processing and consistent quality.
- For long articles, split by paragraph/sentence and stitch the outputs – use `examples/infer_long_text.py`.
- Always obtain consent before cloning someone’s voice.

---

## ⚠️ Troubleshooting

| Issue | Likely cause | How to fix |
|-------|--------------|------------|
| `ValueError: Could not find libespeak...` | eSpeak NG is missing or the path is incorrect | Install eSpeak NG and set `PHONEMIZER_ESPEAK_LIBRARY` if required |
| `401 Unauthorized` when downloading `facebook/w2v-bert-2.0` | Invalid or stale Hugging Face token in the environment | Run `huggingface-cli login --token …` or remove `HF_TOKEN` to use anonymous access |
| `CUDA out of memory` | GPU VRAM is insufficient | Switch to CPU (`backbone_device="cpu"` & `codec_device="cpu"`) or use a quantized checkpoint |
| `No valid speech tokens found` | Prompt too long, empty text, or poor reference clip | Shorten the input, double-check normalization, or pick another reference sample |

---

## 📚 References

- [GitHub Repository](https://github.com/pnnbao97/VieNeu-TTS)  
- [Hugging Face Model Card](https://huggingface.co/pnnbao-ump/VieNeu-TTS)  
- [NeuTTS Air base model](https://huggingface.co/neuphonic/neutts-air)  
- [Fine-tuning guide](https://github.com/pnnbao-ump/VieNeuTTS/blob/main/finetune.ipynb)  
- [VieNeuCodec dataset](https://huggingface.co/datasets/pnnbao-ump/VieNeuCodec-dataset)

---

## 📄 License

Apache License 2.0

---

## 📑 Citation

```bibtex
@misc{vieneutts2025,
  title        = {VieNeu-TTS: Vietnamese Text-to-Speech with Instant Voice Cloning},
  author       = {Pham Nguyen Ngoc Bao},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/pnnbao-ump/VieNeu-TTS}}
}
```

Please also cite the base model:

```bibtex
@misc{neuttsair2025,
  title        = {NeuTTS Air: On-Device Speech Language Model with Instant Voice Cloning},
  author       = {Neuphonic},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/neuphonic/neutts-air}}
}
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository  
2. Create a feature branch: `git checkout -b feature/amazing-feature`  
3. Commit your changes: `git commit -m "Add amazing feature"`  
4. Push the branch: `git push origin feature/amazing-feature`  
5. Open a pull request

---

## 📞 Support

- GitHub Issues: [github.com/pnnbao97/VieNeu-TTS/issues](https://github.com/pnnbao97/VieNeu-TTS/issues)  
- Hugging Face: [huggingface.co/pnnbao-ump](https://huggingface.co/pnnbao-ump)  
- Facebook: [Phạm Nguyễn Ngọc Bảo](https://www.facebook.com/bao.phamnguyenngoc.5)

---

## 🙏 Acknowledgements

This project builds upon [NeuTTS Air](https://huggingface.co/neuphonic/neutts-air) by Neuphonic. Huge thanks to the team for open-sourcing such a powerful base model.

---

**Made with ❤️ for the Vietnamese TTS community**




