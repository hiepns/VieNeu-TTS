[Github](https://github.com/pnnbao97/VieNeu-TTS)
**VieNeu-TTS** là mô hình Text-to-Speech (TTS) tiếng Việt đầu tiên chạy trên thiết bị cá nhân với khả năng nhân bản giọng nói tức thì. Được fine-tune từ [NeuTTS Air](https://huggingface.co/neuphonic/neutts-air), VieNeu-TTS mang đến giọng nói tiếng Việt tự nhiên, siêu chân thực với hiệu suất thời gian thực trên CPU.

Dựa trên backbone Qwen 0.5B LLM, VieNeu-TTS kết hợp giữa tốc độ, kích thước nhỏ gọn và chất lượng âm thanh cao - hoàn hảo cho các ứng dụng voice agent, trợ lý ảo, đồ chơi tương tác và các ứng dụng yêu cầu bảo mật cao chạy trên thiết bị local.

Tác giả: Phạm Nguyễn Ngọc Bảo

## Chi tiết mô hình

VieNeu-TTS được xây dựng dựa trên kiến trúc NeuTTS Air với các thành phần chính:

- **Base Model**: Qwen 0.5B
- **Audio Codec**: NeuCodec
- **Format**: Safetensors và GGUF (Q8, Q4) cho suy luận hiệu quả trên thiết bị
- **Trách nhiệm**: Audio đầu ra có watermark tích hợp
- **Tốc độ suy luận**: Sinh giọng nói thời gian thực trên thiết bị tầm trung
- **Tiêu thụ điện năng**: Tối ưu cho thiết bị di động và nhúng
- **Dataset huấn luyện**: 
  - [VieNeuCodec-dataset](https://huggingface.co/datasets/pnnbao-ump/VieNeuCodec-dataset) - 74.9k mẫu audio tiếng Việt
  - Fine-tuned từ base model đã được train trên [Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)
 
## Bắt đầu

### Clone Git Repo

```bash
git clone https://github.com/pnnbao97/VieNeu-TTS.git
cd VieNeu-TTS
```

### Cài đặt espeak (dependency bắt buộc)

Tham khảo hướng dẫn chi tiết tại: https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md

```bash
# Mac OS
brew install espeak

# Ubuntu/Debian
sudo apt install espeak

# Arch Linux
paru -S aur/espeak
```

### Cài đặt Python dependencies

File requirements bao gồm các dependencies cần thiết để chạy model với PyTorch. Khi sử dụng ONNX decoder hoặc GGML model, một số dependencies (như PyTorch) có thể không cần thiết.

Inference tương thích và đã được test trên python>=3.11.

```bash
pip install -r requirements.txt
```
## Ví dụ cơ bản

```python
from vieneutts import VieNeuTTS
import soundfile as sf
import os

# Văn bản cần tổng hợp
input_texts = [
    "Các khóa học trực tuyến đang giúp học sinh tiếp cận kiến thức mọi lúc mọi nơi.",
    "Các nghiên cứu về bệnh Alzheimer cho thấy tác dụng tích cực của các bài tập trí não.",
    "Một tiểu thuyết trinh thám hiện đại dẫn dắt độc giả qua những tình tiết phức tạp, bí ẩn.",
]

output_dir = "./output_audio"
os.makedirs(output_dir, exist_ok=True)

# Đường dẫn file tham chiếu
# Nam miền Nam
ref_audio_path = "./sample/id_0001.wav"
ref_text_path = "./sample/id_0001.txt"
# Nữ miền Nam
# ref_audio_path = "./sample/id_0002.wav"
# ref_text_path = "./sample/id_0002.txt"

ref_text = open(ref_text_path, "r", encoding="utf-8").read()

# Khởi tạo model
tts = VieNeuTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS",
    backbone_device="cuda",  # hoặc "cpu" nếu không có GPU
    codec_repo="neuphonic/neucodec",
    codec_device="cuda"  # hoặc "cpu" nếu không có GPU
)

print("Encoding reference audio")
ref_codes = tts.encode_reference(ref_audio_path)

# Tổng hợp giọng nói cho nhiều văn bản
for i, text in enumerate(input_texts, 1):
    print(f"Generating audio for example {i}: {text}")
    wav = tts.infer(text, ref_codes, ref_text)
    output_path = os.path.join(output_dir, f"output_{i}.wav")
    sf.write(output_path, wav, 24000)
    print(f"Saved to {output_path}")
```
### Khuyến cáo

Vui lòng không sử dụng mô hình này cho mục đích xấu hoặc vi phạm pháp luật, bao gồm:

- Mạo danh giọng nói người khác mà không có sự đồng ý
- Tạo nội dung sai sự thật, lừa đảo
- Vi phạm quyền riêng tư hoặc quyền sở hữu trí tuệ
- Các hành vi vi phạm pháp luật khác

Hãy tôn trọng quyền riêng tư và quyền sở hữu trí tuệ của người khác.

## Giới hạn

- Mô hình có thể không phát âm chính xác 100% các từ tiếng Việt phức tạp hoặc từ vựng chuyên ngành
- Chất lượng đầu ra phụ thuộc nhiều vào chất lượng của audio tham chiếu
- Hiệu suất có thể giảm với văn bản quá dài (khuyến nghị chia nhỏ văn bản dài)

## License

Apache 2.0

## Trích dẫn

Nếu bạn sử dụng VieNeu-TTS trong nghiên cứu hoặc ứng dụng của mình, vui lòng trích dẫn:

```bibtex
@misc{vieneutts2025,
  title={VieNeu-TTS: Vietnamese Text-to-Speech with Instant Voice Cloning},
  author={Pham Nguyen Ngoc Bao},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/pnnbao-ump/VieNeu-TTS}}
}
```

Và base model NeuTTS Air:

```bibtex
@misc{neuttsair2025,
  title={NeuTTS Air: On-Device Speech Language Model with Instant Voice Cloning},
  author={Neuphonic},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/neuphonic/neutts-air}}
}
```

## Liên hệ

- **GitHub**: [pnnbao97](https://github.com/pnnbao97)
- **Hugging Face**: [pnnbao-ump](https://huggingface.co/pnnbao-ump)
- **Facebook**: [Phạm Nguyễn Ngọc Bảo](https://www.facebook.com/bao.phamnguyenngoc.5)

## Ghi nhận

Dự án này được xây dựng dựa trên [NeuTTS Air](https://huggingface.co/neuphonic/neutts-air) của Neuphonic. Xin gửi lời cảm ơn chân thành đến đội ngũ Neuphonic đã tạo ra mô hình base xuất sắc này và công khai cho cộng đồng.

---

**Lưu ý**: Đây là phiên bản nghiên cứu và thử nghiệm. Vui lòng báo cáo các vấn đề hoặc đóng góp cải tiến qua GitHub Issues.

