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

| Model | Format | Device | Quality | Speed | Streaming | RAM Usage |
|-------|--------|--------|---------|-------|-----------|-----------|
| VieNeu-TTS | PyTorch | GPU/CPU | ⭐⭐⭐⭐⭐ | Medium | ❌ | ~2GB |
| VieNeu-TTS-q8-gguf | GGUF Q8 | CPU/GPU | ⭐⭐⭐⭐ | Fast | ✅ | ~1.5GB |
| VieNeu-TTS-q4-gguf | GGUF Q4 | CPU/GPU | ⭐⭐⭐ | Very Fast | ✅ | ~1GB |

**Recommendations:**
- **GPU users**: Use `VieNeu-TTS` (PyTorch) for best quality
- **CPU users**: Use `VieNeu-TTS-q4-gguf` for fastest inference or `VieNeu-TTS-q8-gguf` for better quality
- **Streaming**: Only GGUF models support streaming inference

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

### 3. Install Python dependencies (Python ≥ 3.12)

```bash
uv sync
```

**Note:** If you plan to use GGUF models with GPU acceleration, you may need to install `llama-cpp-python` with CUDA support:

```bash
# For CUDA support (optional, only if you have NVIDIA GPU)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

---

## 📦 Project Structure

```
VieNeu-TTS/
├── examples/
│   ├── infer_long_text.py     # CLI for long-form synthesis (chunked)
│   └── sample_long_text.txt   # Example paragraph for testing
├── gradio_app.py              # Local Gradio web demo
├── main.py                    # Basic batch inference script
├── config.yaml                # Configuration for models and voices
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
│   ├── core_utils.py          # Utility functions
│   ├── normalize_text.py      # Vietnamese text normalization pipeline
│   ├── phonemize_text.py      # Text to phoneme conversion
│   └── phoneme_dict.json      # Phoneme dictionary
├── vieneu_tts/
│   ├── __init__.py
│   └── vieneu_tts.py          # Core VieNeuTTS implementation
├── web_ui/                    # React-based web UI (optional)
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

## 💻 Sử dụng GGUF Q4 và Q8 cho CPU

GGUF models được tối ưu hóa đặc biệt cho CPU, giúp chạy nhanh hơn và tiết kiệm bộ nhớ so với model PyTorch gốc.

### Cách 1: Sử dụng qua Gradio Web UI

1. **Khởi động Gradio app:**
   ```bash
   uv run gradio_app.py
   ```

2. **Chọn model trong giao diện:**
   - **Backbone**: Chọn `VieNeu-TTS-q4-gguf` (nhanh nhất) hoặc `VieNeu-TTS-q8-gguf` (chất lượng tốt hơn)
   - **Codec**: Chọn `NeuCodec ONNX (Fast CPU)` để tối ưu tốc độ trên CPU
   - **Device**: Chọn `CPU`

3. **Click "🔄 Tải Model"** và đợi model tải về (lần đầu sẽ mất thời gian)

4. **Sử dụng như bình thường** - model sẽ tự động chạy trên CPU

### Cách 2: Sử dụng qua Python code

#### Sử dụng GGUF Q4 (nhẹ nhất, nhanh nhất)

```python
from vieneu_tts import VieNeuTTS
import soundfile as sf
import torch

# Khởi tạo model Q4 cho CPU
tts = VieNeuTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS-q4-gguf",
    backbone_device="cpu",  # Sử dụng CPU
    codec_repo="neuphonic/neucodec-onnx-decoder",  # ONNX codec cho CPU
    codec_device="cpu"
)

# Load reference codes (pre-encoded cho ONNX codec)
ref_codes = torch.load("./sample/Vĩnh (nam miền Nam).pt", map_location="cpu")
with open("./sample/Vĩnh (nam miền Nam).txt", "r", encoding="utf-8") as f:
    ref_text = f.read()

# Tổng hợp giọng nói
text = "Đây là ví dụ sử dụng model Q4 trên CPU."
wav = tts.infer(text, ref_codes, ref_text)

# Lưu file audio
sf.write("output_q4.wav", wav, 24000)
print("✅ Đã tạo file output_q4.wav")
```

#### Sử dụng GGUF Q8 (chất lượng tốt hơn)

```python
from vieneu_tts import VieNeuTTS
import soundfile as sf
import torch

# Khởi tạo model Q8 cho CPU
tts = VieNeuTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS-q8-gguf",
    backbone_device="cpu",
    codec_repo="neuphonic/neucodec-onnx-decoder",
    codec_device="cpu"
)

# Load reference
ref_codes = torch.load("./sample/Vĩnh (nam miền Nam).pt", map_location="cpu")
with open("./sample/Vĩnh (nam miền Nam).txt", "r", encoding="utf-8") as f:
    ref_text = f.read()

# Tổng hợp
text = "Đây là ví dụ sử dụng model Q8 trên CPU với chất lượng tốt hơn."
wav = tts.infer(text, ref_codes, ref_text)

sf.write("output_q8.wav", wav, 24000)
print("✅ Đã tạo file output_q8.wav")
```

### Streaming với GGUF models

GGUF models hỗ trợ streaming inference, cho phép nghe audio trong khi đang tạo. 

### Lưu ý quan trọng khi sử dụng GGUF trên CPU

1. **Pre-encoded codes**: Khi sử dụng `neuphonic/neucodec-onnx-decoder`, bạn cần sử dụng file `.pt` (pre-encoded codes) thay vì encode từ audio. Các file `.pt` đã có sẵn trong thư mục `sample/`.

2. **Nếu không có file .pt**: Bạn có thể encode từ audio bằng cách sử dụng codec PyTorch trước:
   ```python
   # Tạm thời dùng PyTorch codec để encode
   tts_temp = VieNeuTTS(
       backbone_repo="pnnbao-ump/VieNeu-TTS-q4-gguf",
       backbone_device="cpu",
       codec_repo="neuphonic/neucodec",  # PyTorch codec
       codec_device="cpu"
   )
   ref_codes = tts_temp.encode_reference("./sample/Vĩnh (nam miền Nam).wav")
   torch.save(ref_codes, "./sample/Vĩnh (nam miền Nam).pt")
   ```

3. **Tối ưu hiệu năng CPU**:
   - Sử dụng Q4 cho tốc độ tối đa
   - Sử dụng ONNX codec cho codec decoding nhanh hơn
   - Giảm `max_chars_per_chunk` nếu gặp vấn đề về bộ nhớ

4. **GPU acceleration (tùy chọn)**: Nếu có NVIDIA GPU và đã cài `llama-cpp-python` với CUDA, bạn có thể dùng `backbone_device="gpu"` để tăng tốc.

---

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

Each reference voice includes:
- `.wav` - Audio file
- `.txt` - Transcript file
- `.pt` - Pre-encoded codes (for ONNX codec)

**Note:** GGUF models hiện tại chỉ hỗ trợ 4 giọng: Vĩnh, Bình, Ngọc, và Dung.

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
