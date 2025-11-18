# VieNeu-TTS

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/pnnbao97/VieNeu-TTS)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)

<img width="899" height="615" alt="Untitled" src="https://github.com/user-attachments/assets/7eb9b816-6ab7-4049-866f-f85e36cb9c6f" />

**VieNeu-TTS-1000h** is an advanced on-device Vietnamese Text-to-Speech (TTS) model with **instant voice cloning**.  

Trained on ~1000 hours of high-quality Vietnamese speech, this model represents a significant upgrade from VieNeu-TTS-140h with the following improvements:

- **Enhanced pronunciation**: More accurate and stable Vietnamese pronunciation
- **Code-switching support**: Seamless transitions between Vietnamese and English
- **Better voice cloning**: Higher fidelity and speaker consistency
- **Real-time synthesis**: 24 kHz waveform generation on CPU or GPU

Fine-tuned from **NeuTTS Air**, VieNeu-TTS-1000h delivers production-ready speech synthesis fully offline.

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

## 💝 Support This Project

**VieNeu-TTS** is a free, open-source project. However, training high-quality TTS models on **1000+ hours of speech data** requires significant computational resources.

If you find this project useful, please consider supporting its development:

<div align="center">

[![Buy Me a Coffee](https://img.shields.io/badge/☕_Buy_Me_a_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/pnnbao)

</div>

**Your support helps:**

- 💰 **GPU Training Costs**: Training on 1000+ hours costs thousands of dollars in compute
- 🚀 **New Features**: Emotion control, speaking styles, GGUF quantization
- 📊 **Dataset Expansion**: Collecting more diverse Vietnamese voices (North, Central, South)
- 🎯 **Quality Improvements**: Better pronunciation, naturalness, and voice cloning fidelity
- 🌍 **Bilingual Support**: Vietnamese + English code-switching capabilities
- 🔧 **Maintenance**: Bug fixes, updates, and community support

<div align="center">

*Every contribution, big or small, makes a real difference!*  
*Thank you for supporting Vietnamese AI development!* 🇻🇳🙏

</div>

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

## Quick Usage (Python)

```python
from vieneu_tts import VieNeuTTS
import soundfile as sf
import os

input_texts = [
    "Các khóa học trực tuyến đang giúp học sinh tiếp cận kiến thức mọi lúc mọi nơi. Giáo viên sử dụng video, bài tập tương tác và thảo luận trực tuyến để nâng cao hiệu quả học tập.",

    "Các nghiên cứu về bệnh Alzheimer cho thấy tác dụng tích cực của các bài tập trí não và chế độ dinh dưỡng lành mạnh, giúp giảm tốc độ suy giảm trí nhớ ở người cao tuổi.",

    "Một tiểu thuyết trinh thám hiện đại dẫn dắt độc giả qua những tình tiết phức tạp, bí ẩn, kết hợp yếu tố tâm lý sâu sắc khiến người đọc luôn hồi hộp theo dõi diễn biến câu chuyện.",

    "Các nhà khoa học nghiên cứu gen người phát hiện những đột biến mới liên quan đến bệnh di truyền. Điều này giúp nâng cao khả năng chẩn đoán và điều trị.",
]

output_dir = "./output_audio"
os.makedirs(output_dir, exist_ok=True)

def main(backbone="pnnbao-ump/VieNeu-TTS-1000h", codec="neuphonic/neucodec"):
    """
    In the sample directory, there are 7 wav files and 7 txt files with matching names.
    These are pre-prepared reference files for testing:
    - id_0001.wav + id_0001.txt
    - id_0002.wav + id_0002.txt
    - id_0003.wav + id_0003.txt
    - id_0004.wav + id_0004.txt
    - id_0005.wav + id_0005.txt
    - id_0006.wav + id_0006.txt
    - id_0007.wav + id_0007.txt
    
    Odd numbers = Male voices
    Even numbers = Female voices
    
    Note: The model can clone any voice you provide (with corresponding text).
    However, quality may not match the sample files. For best results, finetune
    the model on your target voice. See finetune guide at:
    https://github.com/pnnbao-ump/VieNeuTTS/blob/main/finetune.ipynb
    """
    # Male voice (South accent)
    ref_audio_path = "./sample/id_0001.wav"
    ref_text_path = "./sample/id_0001.txt"
    
    # Female voice (South accent) - uncomment to use
    # ref_audio_path = "./sample/id_0002.wav"
    # ref_text_path = "./sample/id_0002.txt"

    ref_text_raw = open(ref_text_path, "r", encoding="utf-8").read()
    
    if not ref_audio_path or not ref_text_raw:
        print("No reference audio or text provided.")
        return None

    # Initialize VieNeuTTS-1000h
    tts = VieNeuTTS(
        backbone_repo=backbone,
        backbone_device="cuda",
        codec_repo=codec,
        codec_device="cuda"
    )

    print("Encoding reference audio...")
    ref_codes = tts.encode_reference(ref_audio_path)

    # Generate speech for all input texts
    for i, text in enumerate(input_texts, 1):
        print(f"Generating audio {i}/{len(input_texts)}: {text[:50]}...")
        wav = tts.infer(text, ref_codes, ref_text_raw)
        output_path = os.path.join(output_dir, f"output_{i}.wav")
        sf.write(output_path, wav, 24000)
        print(f"✓ Saved to {output_path}")

if __name__ == "__main__":
    main()
```

### CLI example (`main.py`)

```bash
python main.py
```

This script runs several normalized sentences using the bundled sample voice and writes `output_*.wav` files under `output_audio/`.

### Gradio web demo
[<img width="600" height="595" alt="VieNeu-TTS" src="https://github.com/user-attachments/assets/66c098c4-d184-4e7a-826a-ba8c6c556fab" />](https://github.com/user-attachments/assets/5ad53bc9-e816-41a7-9474-ea470b1cbfdd)

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








