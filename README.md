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
- **Multiple model formats**: Support for PyTorch, GGUF Q4/Q8 (CPU optimized), and ONNX codec

VieNeu-TTS-1000h delivers production-ready speech synthesis fully offline.

**Author:** Phạm Nguyễn Ngọc Bảo

[<img width="600" height="595" alt="VieNeu-TTS" src="https://github.com/user-attachments/assets/6b32df9d-7e2e-474f-94c8-43d6fa586d15" />](https://github.com/user-attachments/assets/6b32df9d-7e2e-474f-94c8-43d6fa586d15)

---

## 🔬 Model Overview

- **Backbone:** Qwen 0.5B LLM (chat template)
- **Audio codec:** NeuCodec (torch implementation; ONNX & quantized variants supported)
- **Context window:** 2 048 tokens shared by prompt text and speech tokens
- **Output watermark:** Enabled by default
- **Training data:**  
  - [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h) — 443,641 curated Vietnamese samples  

### Model Variants

| Model | Format | Device | Quality | Speed | Streaming |
|-------|--------|--------|---------|-------|-----------|
| VieNeu-TTS | PyTorch | GPU/CPU | ⭐⭐⭐⭐⭐ | Very Fast with lmdeploy | ❌ |
| VieNeu-TTS-q8-gguf | GGUF Q8 | CPU/GPU | ⭐⭐⭐⭐ | Fast | ✅ |
| VieNeu-TTS-q4-gguf | GGUF Q4 | CPU/GPU | ⭐⭐⭐ | Very Fast | ✅ |

**Recommendations:**
- **GPU users**: Use `VieNeu-TTS` (PyTorch) for best quality
- **CPU users**: Use `VieNeu-TTS-q4-gguf` for fastest inference or `VieNeu-TTS-q8-gguf` for better quality
- **Streaming**: Only GGUF models support streaming inference

---

## ✅ Todo & Status

- [x] Publish safetensor artifacts
- [x] Release GGUF Q4 / Q8 models
- [x] Release datasets (1000h and 140h)
- [x] Enable streaming on GPU
- [ ] Provide Dockerized setup
- [ ] Release fine-tuning code

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

### 3. Install Python dependencies (Python ≥ 3.12)

```bash
uv sync
```

**Optional dependencies:**

- **For GGUF models with CPU:** Install `llama-cpp-python` with CPU support:
  ```bash
  uv pip install llama-cpp-python==0.3.2 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu/
  ```

- **For GGUF models with GPU:** Install `llama-cpp-python` with CUDA support:
  ```bash
  CMAKE_ARGS="-DLLAMA_CUBLAS=on" uv pip install llama-cpp-python --force-reinstall --no-cache-dir
  ```
  
- **For LMDeploy optimizations (GPU only):** Install `lmdeploy` for faster GPU inference:
  ```bash
  uv pip install lmdeploy
  uv pip install triton
  ```
  For Windows:
  ```bash
  uv pip install lmdeploy
  uv pip install triton-windows 
  ```
  This enables batch processing, Triton compilation, and KV cache quantization in the Gradio app.

---

## 🐋 Docker Deployment

For a quick start or production deployment without manually installing dependencies, use Docker.

### Quick Start

```bash
# Run with CPU
docker compose --profile cpu up

# Run with GPU (requires NVIDIA Container Toolkit)
docker compose --profile gpu up
```

Access the Web UI at `http://localhost:7860`.

For detailed deployment instructions, including production setup, see [Deploy.md](Deploy.md).

---

## 📦 Project Structure

```
VieNeu-TTS/
├── examples/
│   ├── infer_long_text.py     # CLI for long-form synthesis (chunked)
│   └── sample_long_text.txt   # Example paragraph for testing
├── gradio_app.py              # Local Gradio web demo with LMDeploy support
├── main.py                    # Basic batch inference script
├── config.yaml                # Configuration for models, codecs, and voices
├── output_audio/              # Generated audio (created when running scripts)
├── sample/                    # Reference voices (audio + transcript + codes)
│   ├── Bình (nam miền Bắc).wav/txt/pt
│   ├── Đoan (nữ miền Nam).wav/txt/pt
│   ├── Dung (nữ miền Nam).wav/txt/pt
│   ├── Hương (nữ miền Bắc).wav/txt/pt
│   ├── Ly (nữ miền Bắc).wav/txt/pt
│   ├── Ngọc (nữ miền Bắc).wav/txt/pt
│   ├── Nguyên (nam miền Nam).wav/txt/pt
│   ├── Sơn (nam miền Nam).wav/txt/pt
│   ├── Tuyên (nam miền Bắc).wav/txt/pt
│   └── Vĩnh (nam miền Nam).wav/txt/pt
├── utils/
│   ├── __init__.py
│   ├── core_utils.py          # Text chunking utilities
│   ├── normalize_text.py      # Vietnamese text normalization pipeline
│   ├── phonemize_text.py      # Text to phoneme conversion
│   └── phoneme_dict.json      # Phoneme dictionary
├── vieneu_tts/
│   ├── __init__.py            # Exports VieNeuTTS and FastVieNeuTTS
│   └── vieneu_tts.py          # Core VieNeuTTS implementation (VieNeuTTS & FastVieNeuTTS)
├── README.md
├── requirements.txt           # Basic dependencies (legacy)
├── pyproject.toml             # Project configuration with full dependencies (UV)
└── uv.lock                    # UV lock file for dependency management
```

---

## 🚀 Quickstart

### Gradio web demo

```bash
uv run gradio_app.py
```

Then open `http://127.0.0.1:7860` to:

- Choose from multiple model variants (PyTorch, GGUF Q4/Q8)
- Pick one of ten reference voices (5 male, 5 female; North and South accents)
- Upload your own reference audio + transcript
- Enter text up to 3000 characters (with chunking support)
- Preview or download the synthesized audio

### Basic Python usage

```python
from vieneu_tts import VieNeuTTS
import soundfile as sf

# Initialize with GGUF Q4 model for CPU
tts = VieNeuTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS-q4-gguf",
    backbone_device="cpu",
    codec_repo="neuphonic/neucodec-onnx-decoder",
    codec_device="cpu"
)

# Load reference (using pre-encoded codes for ONNX codec)
import torch
ref_codes = torch.load("./sample/Vĩnh (nam miền Nam).pt", map_location="cpu")
with open("./sample/Vĩnh (nam miền Nam).txt", "r", encoding="utf-8") as f:
    ref_text = f.read()

# Generate speech
text = "Xin chào, đây là một ví dụ về tổng hợp giọng nói tiếng Việt."
wav = tts.infer(text, ref_codes, ref_text)

# Save audio
sf.write("output.wav", wav, 24000)
```

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





