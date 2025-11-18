import gradio as gr
import soundfile as sf
import tempfile
import torch
from vieneu_tts import VieNeuTTS

print("⏳ Đang khởi động VieNeu-TTS...")

# Khởi tạo model
print("📦 Đang tải model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Sử dụng thiết bị: {device.upper()}")

tts = VieNeuTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS-1000h",
    backbone_device=device,
    codec_repo="neuphonic/neucodec",
    codec_device=device
)
print("✅ Model đã tải xong!")

# Danh sách giọng mẫu
VOICE_SAMPLES = {
    "Nam 1": {
        "audio": "./sample/id_0001.wav",
        "text": "./sample/id_0001.txt"
    },
    "Nữ 1": {
        "audio": "./sample/id_0002.wav",
        "text": "./sample/id_0002.txt"
    },
    "Nam 2": {
        "audio": "./sample/id_0003.wav",
        "text": "./sample/id_0003.txt"
    },
    "Nữ 2": {
        "audio": "./sample/id_0004.wav",
        "text": "./sample/id_0004.txt"
    },
    "Nam 3": {
        "audio": "./sample/id_0005.wav",
        "text": "./sample/id_0005.txt"
    },
    "Nam 4": {
        "audio": "./sample/id_0007.wav",
        "text": "./sample/id_0007.txt"
    }
}

def synthesize_speech(text, voice_choice, custom_audio=None, custom_text=None):
    """Tổng hợp giọng nói từ văn bản"""
    try:
        if not text or text.strip() == "":
            return None, "❌ Vui lòng nhập văn bản cần tổng hợp"
        
        if len(text) > 250:
            return None, "❌ Văn bản quá dài! Vui lòng nhập tối đa 250 ký tự. Để tổng hợp văn bản dài hơn, vui lòng tham khảo examples/infer_long_text.py"
        
        # Xác định reference audio và text
        if custom_audio is not None and custom_text:
            ref_audio_path = custom_audio
            ref_text_raw = custom_text
            print("🎨 Sử dụng giọng tùy chỉnh")
        elif voice_choice in VOICE_SAMPLES:
            ref_audio_path = VOICE_SAMPLES[voice_choice]["audio"]
            ref_text_path = VOICE_SAMPLES[voice_choice]["text"]
            with open(ref_text_path, "r", encoding="utf-8") as f:
                ref_text_raw = f.read()
            print(f"🎤 Sử dụng giọng: {voice_choice}")
        else:
            return None, "❌ Vui lòng chọn giọng hoặc tải lên audio tùy chỉnh"
        
        # Encode và tổng hợp
        print(f"📝 Đang xử lý: {text[:50]}...")
        ref_codes = tts.encode_reference(ref_audio_path)
        
        print(f"🎵 Đang tổng hợp giọng nói trên {device.upper()}...")
        wav = tts.infer(text, ref_codes, ref_text_raw)
        
        # Lưu file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, wav, 24000)
            output_path = tmp_file.name
        
        print("✅ Hoàn thành!")
        return output_path, f"✅ Tổng hợp thành công"
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"❌ Lỗi: {str(e)}"

# Custom CSS - tối giản
custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}
.warning-box {
    background-color: #fef3c7;
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 10px 0;
    color: #000000;
}
"""

# Tạo giao diện
with gr.Blocks(title="VieNeu-TTS", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""


    # VieNeu-TTS

    Hệ thống tổng hợp tiếng nói tiếng Việt sử dụng Large Language Model

    **Phiên bản:** VieNeu-TTS-1000h (model mới nhất, train trên 1000 giờ dữ liệu)

    [GitHub](https://github.com/pnnbao97/VieNeu-TTS) • [Model Card](https://huggingface.co/pnnbao-ump/VieNeu-TTS) • [Finetune Guide](https://github.com/pnnbao-ump/VieNeuTTS/blob/main/finetune.ipynb)
    
    """)
    # Main interface
    with gr.Row():

        
        with gr.Column(scale=1):
            
            text_input = gr.Textbox(
                label="Văn bản",
                placeholder="Nhập văn bản tiếng Việt (khuyến cáo dưới 250 ký tự)...",
                lines=5,
                value="Trí tuệ nhân tạo đang cách mạng hóa nhiều lĩnh vực, từ y tế, giáo dục đến giao thông vận tải, mang lại những giải pháp thông minh và hiệu quả."
            )
            
            char_count = gr.Markdown("**142 / 250 ký tự**")
            
            voice_select = gr.Radio(
                choices=list(VOICE_SAMPLES.keys()),
                label="Chọn giọng",
                value="Nam 1"
            )
            
            submit_btn = gr.Button("Tổng hợp", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Kết quả", type="filepath")
            status_output = gr.Textbox(label="Trạng thái", interactive=False, show_label=False)
            
            with gr.Accordion("Giọng tùy chỉnh", open=False):
                gr.Markdown("""
Tải lên file audio và nhập nội dung tương ứng. Để có kết quả tốt nhất, nên finetune model trên giọng của bạn.
                """)
                custom_audio = gr.Audio(label="File audio (.wav)", type="filepath")
                custom_text = gr.Textbox(
                    label="Nội dung audio",
                    placeholder="Nhập chính xác nội dung...",
                    lines=2
                )
            gr.HTML("""
            <div class="warning-box" style="color: #000000;">
                ⚠️ Chúng tôi khuyến cáo sử dụng đoạn văn bản <250 ký tự để đảm bảo chất lượng tốt nhất. 
                Nếu muốn tổng hợp văn bản dài hơn, vui lòng tham khảo code trong examples/infer_long_text.py
            </div>
            """)
    
    # Examples
    with gr.Row():
        gr.Examples(
            examples=[
                ["Trí tuệ nhân tạo đang cách mạng hóa nhiều lĩnh vực, từ y tế, giáo dục đến giao thông vận tải, mang lại những giải pháp thông minh và hiệu quả.", "Nam 1"],
                ["Trên bầu trời xanh thẳm, những đám mây trắng lửng lờ trôi như những chiếc thuyền nhỏ đang lướt nhẹ theo dòng gió. Dưới mặt đất, cánh đồng lúa vàng rực trải dài tới tận chân trời, những bông lúa nghiêng mình theo từng làn gió.", "Nữ 2"],
                ["Legacy là một bộ phim đột phá về mặt âm nhạc, quay phim, hiệu ứng đặc biệt, và tôi rất mừng vì cuối cùng nó cũng được cả giới phê bình lẫn người hâm mộ đánh giá lại. Chúng ta đã quá bất công với bộ phim này vào năm 2010.", "Nam 4"],
                ["Thật đáng ngạc nhiên! Mặc dù con đường này rất xa và khó đi, nhưng với sự kiên trì và sự đồng lòng của tất cả mọi người, chúng ta đã hoàn thành được công việc sửa chữa trước 3 ngày so với kế hoạch ban đầu, bạn có tin không?", "Nữ 1"],
                ["Các bác sĩ đang nghiên cứu một loại vaccine mới chống lại virus cúm mùa. Thí nghiệm lâm sàng cho thấy phản ứng miễn dịch mạnh mẽ và ít tác dụng phụ.", "Nam 2"],
            ],
            inputs=[text_input, voice_select],
            outputs=[audio_output, status_output],
            fn=synthesize_speech,
            cache_examples=False
        )
    
    # Footer info
    gr.Markdown("""
---

**Tác giả:** Phạm Nguyễn Ngọc Bảo • **Model:** VieNeu-TTS-1000h

**Lưu ý:** Nếu muốn sử dụng model cũ VieNeu-TTS-140h, hãy thay đổi `backbone_repo` trong mã nguồn

---

### Ủng hộ dự án

VieNeu-TTS là dự án miễn phí và mã nguồn mở. Tuy nhiên, việc train model TTS chất lượng cao trên 1000+ giờ dữ liệu đòi hỏi nguồn lực tính toán đáng kể.

Nếu bạn thấy dự án này hữu ích, hãy cân nhắc ủng hộ:

☕ [Buy Me a Coffee](https://buymeacoffee.com/pnnbao)

    """)
    
    # Update character count
    def update_char_count(text):
        count = len(text) if text else 0
        color = "#dc2626" if count > 250 else "#374151"
        return f"<span style='color: {color}; font-weight: 500'>{count} / 250 ký tự</span>"
    
    text_input.change(
        fn=update_char_count,
        inputs=[text_input],
        outputs=[char_count]
    )
    
    # Event handler
    submit_btn.click(
        fn=synthesize_speech,
        inputs=[text_input, voice_select, custom_audio, custom_text],
        outputs=[audio_output, status_output]
    )

# Launch
if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )