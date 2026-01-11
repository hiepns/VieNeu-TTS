# 🎯 Hướng dẫn sử dụng LoRA Adapter trong VieNeu-TTS

## 📖 Giới thiệu

Tab "LoRA Adapter" cho phép bạn sử dụng các mô hình giọng nói đã được fine-tune bằng LoRA (Low-Rank Adaptation) từ HuggingFace.

## 🚀 Cách sử dụng

### Bước 1: Tải Model cơ bản
1. Chọn **Backbone** phù hợp (vd: `VieNeu-TTS-0.3B (GPU)`)
2. Chọn **Codec** (vd: `NeuCodec (Distill)`)
3. **BỎ TICK** "🚀 Optimize with LMDeploy" (LoRA không tương thích với LMDeploy)
4. Click **🔄 Tải Model**

### Bước 2: Chuyển sang tab "🎯 LoRA Adapter"

### Bước 3: Nhập thông tin LoRA
- **HuggingFace Repo ID**: Nhập repo ID của LoRA adapter  
  Ví dụ: `pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen`
  
- **HF Token** (tùy chọn): Chỉ cần điền nếu repo là private
  - Lấy token tại: https://huggingface.co/settings/tokens
  - Để trống nếu repo là public

### Bước 4: Upload Audio Reference
- **Audio reference**: Upload file audio từ tập train của LoRA (3-15 giây)
- **Text tương ứng**: Nhập chính xác nội dung của audio (kể cả dấu câu)

⚠️ **Lưu ý quan trọng:**
- Audio reference **phải** là một trong các audio đã dùng để train LoRA
- Text phải khớp **chính xác 100%** với nội dung audio (kể cả dấu câu .,?!)
- LoRA adapter phải tương thích với backbone đã chọn

### Bước 5: Tổng hợp giọng nói
1. Nhập văn bản cần tổng hợp
2. Click **🎵 Bắt đầu**
3. Hệ thống sẽ:
   - Tải base model
   - Tải và merge LoRA adapter
   - Tổng hợp giọng nói
   - Tự động cleanup và restore model gốc

## 📝 Ví dụ

```
Repo ID: pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen
HF Token: (để trống nếu public)
Audio: ngochuyen_00123.wav (từ tập train)
Text: "Hà Nội mùa thu đẹp lắm."
```

## 🔧 Khắc phục sự cố

### Lỗi "LoRA không hỗ trợ LMDeploy"
- Bỏ tick "🚀 Optimize with LMDeploy" 
- Reload model
- LoRA sẽ chạy với standard PyTorch backend (chậm hơn nhưng vẫn ổn)

### Lỗi "Failed to load LoRA"
- Kiểm tra Repo ID có đúng không
- Nếu repo private, hãy thêm HF Token
- Đảm bảo LoRA tương thích với backbone

### Lỗi "Out of Memory"
- LoRA cần RAM/VRAM để merge
- Thử giảm batch size
- Sử dụng backbone nhỏ hơn (0.3B thay vì 0.5B)

### Audio chất lượng kém
- Đảm bảo audio reference từ tập train
- Text reference phải khớp chính xác
- Thử giọng reference khác từ tập train

## 💡 Tips

1. **Audio reference tốt nhất**: Chọn audio có chất lượng cao, rõ ràng
2. **Text chính xác**: Viết đúng chính tả, dấu câu
3. **Tương thích**: Đảm bảo LoRA được train trên cùng base model
4. **RAM/VRAM**: LoRA cần thêm 1-2GB VRAM khi merge
5. **⚡ Tốc độ**: Model sẽ tự động giữ LoRA đã load trong bộ nhớ. Lần nhấn nút "Bắt đầu" thứ 2 trở đi sẽ không tốn thời gian load lại LoRA, giúp tốc độ sinh giọng nhanh như model gốc. (Trừ khi bạn đổi Repo ID khác hoặc chuyển tab).

## 📚 Tài liệu tham khảo

- [Fine-tune VieNeu-TTS với LoRA](../finetune/README.md)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
