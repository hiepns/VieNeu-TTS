# 🦜 Hướng dẫn Fine-tune VieNeu-TTS (LoRA)

Thư mục này chứa toàn bộ công cụ cần thiết để bạn huấn luyện (fine-tune) mô hình VieNeu-TTS với giọng nói của riêng mình bằng phương pháp **LoRA (Low-Rank Adaptation)**.

## ⚙️ Cài đặt (Setup)

Nếu bạn chưa có sẵn mã nguồn, hãy thực hiện cài đặt môi trường:

```bash
git clone https://github.com/pnnbao97/VieNeu-TTS.git
cd VieNeu-TTS
uv sync
```

## 📋 Quy trình huấn luyện (Workflow)

Để đạt được kết quả tốt nhất, bạn cần đi qua các bước sau:

### 1. Chuẩn bị dữ liệu (`dataset/`)
Bạn cần chuẩn bị:
- Thư mục `finetune/dataset/raw_audio/`: Chứa các file âm thanh (.wav) của người nói. Độ dài mỗi file nên trong khoảng từ 3-15 giây để chất lượng finetune đạt tối đa. Theo kinh nghiệm của chúng tôi, tổng thời lượng nên trong khoảng từ 2-4 giờ để model có thể học hết các đặc điểm của giọng mẫu.
- File `finetune/dataset/metadata.csv`: Chứa thông tin văn bản tương ứng với audio. Định dạng: `file_name|text` (ví dụ: `audio_001.wav|Xin chào Việt Nam.`).

*Mẹo: Nếu chưa có dữ liệu, bạn có thể chạy `uv run python finetune/data_scripts/get_hf_sample.py` để tải dữ liệu mẫu.*

### 2. Tiền xử lý và Làm sạch dữ liệu
Chạy các script sau theo thứ tự:

1.  **Lọc dữ liệu (`filter_data.py`)**: Loại bỏ các đoạn âm thanh quá ngắn, quá dài hoặc văn bản chứa ký tự không hợp lệ.
    ```bash
    uv run python finetune/data_scripts/filter_data.py
    ```
    *Kết quả: Tạo ra file `metadata_cleaned.csv`.*

2.  **Mã hóa âm thanh (`encode_data.py`)**: Chuyển đổi audio sang dạng mã hóa của NeuCodec để mô hình LLM có thể học được.
    ```bash
    uv run python finetune/data_scripts/encode_data.py
    ```
    *Kết quả: Tạo ra file `metadata_encoded.csv`.*

### 3. Cấu hình huấn luyện (`configs/lora_config.py`)
Mở file `finetune/configs/lora_config.py` để điều chỉnh các thông số:
- `model`: Chọn base model (vd: `pnnbao-ump/VieNeu-TTS-0.3B`).
- `max_steps`: Số bước huấn luyện (mặc định 5000 là đủ cho giọng đơn lẻ).
- `learning_rate`: Tốc độ học (mặc định là `2e-4`).

### 4. Bắt đầu Huấn luyện (`train.py`)
Chạy script huấn luyện chính:
```bash
uv run python finetune/train.py
```
Mô hình sẽ được lưu định kỳ vào thư mục `finetune/output/`.

---

## 📓 Sử dụng Notebook (Khuyên dùng)
Nếu bạn không quen sử dụng script console, chúng tôi cung cấp file Notebook `finetune_VieNeu-TTS.ipynb`. File này đã tích hợp sẵn mọi bước từ chuẩn bị đến huấn luyện, cực kỳ dễ theo dõi trên Google Colab hoặc máy cục bộ.

---

## 🚀 Sử dụng LoRA sau khi huấn luyện

Sau khi huấn luyện xong, bạn sẽ có các file adapter (vd: `adapter_model.bin`). Bạn có thể:

1.  **Sử dụng trực tiếp trong Gradio**: 
    - Upload thư mục kết quả trong `output/` lên HuggingFace.
    - Nhập Repo ID vào tab **LoRA Adapter** trong ứng dụng Gradio.
2.  **Sử dụng trong Code**:
    ```python
    tts.load_lora_adapter("path/to/your/lora_folder")
    ```

---

## 🦜 Bí kíp để giọng nói hay (Tips)

1.  **Chất lượng Audio**: Đây là yếu tố quan trọng nhất. Audio phải sạch, không có tiếng vang (reverb), không có nhạc nền hoặc tiếng ồn.
2.  **Nội dung đa dạng**: Cố gắng có đa dạng các loại câu (câu hỏi, câu cảm thán, câu khẳng định) để mô hình học được biểu cảm.
3.  **Dấu câu chính xác**: Hãy đảm bảo văn bản trong `metadata.csv` khớp 100% với những gì người nói phát âm, kể cả các dấu ngắt nghỉ.
4.  **Hardware**: Khuyên dùng GPU có bộ nhớ từ 12GB VRAM trở lên (như RTX 3060, 4060 Ti).

---
