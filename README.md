# VieNeu-TTS

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/pnnbao97/VieNeu-TTS)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)

<img width="899" height="615" alt="Untitled" src="https://github.com/user-attachments/assets/7eb9b816-6ab7-4049-866f-f85e36cb9c6f" />

**VieNeu-TTS** is an advanced on-device Vietnamese Text-to-Speech (TTS) model with **instant voice cloning**.  

Trained on ~1000 hours of high-quality Vietnamese speech, this model represents a significant upgrade from VieNeu-TTS-140h with the following improvements:

- **Enhanced pronunciation**: More accurate and stable Vietnamese pronunciation
- **Code-switching support**: Seamless transitions between Vietnamese and English
- **Better voice cloning**: Higher fidelity and speaker consistency
- **Real-time synthesis**: 24 kHz waveform generation on CPU or GPU

VieNeu-TTS-1000h delivers production-ready speech synthesis fully offline.

**Author:** Phạm Nguyễn Ngọc Bảo

[<img width="600" height="595" alt="VieNeu-TTS" src="https://github.com/user-attachments/assets/6b32df9d-7e2e-474f-94c8-43d6fa586d15" />](https://github.com/user-attachments/assets/6b32df9d-7e2e-474f-94c8-43d6fa586d15)

---

## 🔬 Model Overview

- **Backbone:** Qwen 0.5B LLM (chat template)
- **Audio codec:** NeuCodec (torch implementation; ONNX & quantized variants supported)
- **Context window:** 2 048 tokens shared by prompt text and speech tokens
- **Output watermark:** Enabled by default
- **Training data:**  
  - [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h) — 443,641 curated Vietnamese samples  

---

## 🏁 Getting Started

> **📺 Hướng dẫn cài đặt bằng tiếng Việt**: Xem video chi tiết tại [Facebook Reel](https://www.facebook.com/reel/1362972618623766)  

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
uv sync
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
│   ├── Bình (nam miền Bắc).wav/txt
│   ├── Đoan (nữ miền Nam).wav/txt
│   ├── Dung (nữ miền Nam).wav/txt
│   ├── Hương (nữ miền Bắc).wav/txt
│   ├── Ly (nữ miền Bắc).wav/txt
│   ├── Ngọc (nữ miền Bắc).wav/txt
│   ├── Nguyên (nam miền Nam).wav/txt
│   ├── Sơn (nam miền Nam).wav/txt
│   ├── Tuyên (nam miền Bắc).wav/txt
│   └── Vĩnh (nam miền Nam).wav/txt
├── utils/
│   ├── __init__.py
│   ├── normalize_text.py      # Vietnamese text normalization pipeline
│   ├── phonemize_text.py      # Text to phoneme conversion
│   └── phoneme_dict.json      # Phoneme dictionary
├── vieneu_tts/
│   ├── __init__.py
│   └── vieneu_tts.py          # Core VieNeuTTS implementation
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## 🚀 Quickstart

### Gradio web demo

```bash
uv run gradio_app.py
```

Then open `http://127.0.0.1:7860` to:

- Pick one of ten reference voices (5 male, 5 female; North and South accents)
- Upload your own reference audio + transcript
- Enter up to 250 characters per request (recommended)
- Preview or download the synthesized audio

## 🔈 Reference Voices (`sample/`)

| File                    | Gender | Accent | Description        |
|-------------------------|--------|--------|--------------------|
| Bình (nam miền Bắc)     | Male   | North  | Male voice, North accent |
| Tuyên (nam miền Bắc)    | Male   | North  | Male voice, North accent |
| Nguyên (nam miền Nam)   | Male   | South  | Male voice, South accent |
| Sơn (nam miền Nam)      | Male   | South  | Male voice, South accent |
| Vĩnh (nam miền Nam)     | Male   | South  | Male voice, South accent |
| Hương (nữ miền Bắc)     | Female | North  | Female voice, North accent |
| Ly (nữ miền Bắc)        | Female | North  | Female voice, North accent |
| Ngọc (nữ miền Bắc)      | Female | North  | Female voice, North accent |
| Đoan (nữ miền Nam)      | Female | South  | Female voice, South accent |
| Dung (nữ miền Nam)      | Female | South  | Female voice, South accent |

Each reference voice includes both a `.wav` audio file and a matching `.txt` transcript file.

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
