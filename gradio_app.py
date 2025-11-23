import gradio as gr
import soundfile as sf
import tempfile
import torch
from vieneu_tts import VieNeuTTS
import os
import time

print("⏳ Đang khởi động VieNeu-TTS...")

# --- 1. SETUP MODEL ---
print("📦 Đang tải model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Sử dụng thiết bị: {device.upper()}")

try:
    tts = VieNeuTTS(
        backbone_repo="pnnbao-ump/VieNeu-TTS",
        backbone_device=device,
        codec_repo="neuphonic/neucodec",
        codec_device=device
    )
    print("✅ Model đã tải xong!")
except Exception as e:
    print(f"⚠️ Không thể tải model (Chế độ UI Demo): {e}")
    class MockTTS:
        def encode_reference(self, path): return None
        def infer(self, text, ref, ref_text): 
            import numpy as np
            # Giả lập độ trễ để test tính năng đo thời gian
            time.sleep(1.5) 
            return np.random.uniform(-0.5, 0.5, 24000*3)
    tts = MockTTS()

# --- 2. DATA ---
VOICE_SAMPLES = {
    "Bình (nam miền Bắc)": {"audio": "./sample/Bình (nam miền Bắc).wav", "text": "./sample/Bình (nam miền Bắc).txt"},
    "Vĩnh (nam miền Nam)": {"audio": "./sample/Vĩnh (nam miền Nam).wav", "text": "./sample/Vĩnh (nam miền Nam).txt"},
    "Tuyên (nam miền Bắc)": {"audio": "./sample/Tuyên (nam miền Bắc).wav", "text": "./sample/Tuyên (nam miền Bắc).txt"},
    "Nguyên (nam miền Nam)": {"audio": "./sample/Nguyên (nam miền Nam).wav", "text": "./sample/Nguyên (nam miền Nam).txt"},
    "Sơn (nam miền Nam)": {"audio": "./sample/Sơn (nam miền Nam).wav", "text": "./sample/Sơn (nam miền Nam).txt"},
    "Hương (nữ miền Bắc)": {"audio": "./sample/Hương (nữ miền Bắc).wav", "text": "./sample/Hương (nữ miền Bắc).txt"},
    "Ly (nữ miền Bắc)": {"audio": "./sample/Ly (nữ miền Bắc).wav", "text": "./sample/Ly (nữ miền Bắc).txt"},
    "Ngọc (nữ miền Bắc)": {"audio": "./sample/Ngọc (nữ miền Bắc).wav", "text": "./sample/Ngọc (nữ miền Bắc).txt"},
    "Đoan (nữ miền Nam)": {"audio": "./sample/Đoan (nữ miền Nam).wav", "text": "./sample/Đoan (nữ miền Nam).txt"},
    "Dung (nữ miền Nam)": {"audio": "./sample/Dung (nữ miền Nam).wav", "text": "./sample/Dung (nữ miền Nam).txt"}
}

# --- 3. HELPER FUNCTIONS ---
def load_reference_info(voice_choice):
    if voice_choice in VOICE_SAMPLES:
        audio_path = VOICE_SAMPLES[voice_choice]["audio"]
        text_path = VOICE_SAMPLES[voice_choice]["text"]
        try:
            if os.path.exists(text_path):
                with open(text_path, "r", encoding="utf-8") as f:
                    ref_text = f.read()
                return audio_path, ref_text
            else:
                return audio_path, "⚠️ Không tìm thấy file text mẫu."
        except Exception as e:
            return None, f"❌ Lỗi: {str(e)}"
    return None, ""

def synthesize_speech(text, voice_choice, custom_audio, custom_text, mode_tab):
    try:
        if not text or text.strip() == "":
            return None, "⚠️ Vui lòng nhập văn bản cần tổng hợp!"
        
        # --- LOGIC CHECK LIMIT 250 ---
        if len(text) > 250:
            return None, f"❌ Văn bản quá dài ({len(text)}/250 ký tự)! Vui lòng cắt ngắn lại để đảm bảo chất lượng."

        # Logic chọn Reference
        if mode_tab == "custom_mode": 
            if custom_audio is None or not custom_text:
                return None, "⚠️ Vui lòng tải lên Audio và nhập nội dung Audio đó."
            ref_audio_path = custom_audio
            ref_text_raw = custom_text
            print("🎨 Mode: Custom Voice")
        else: # Preset
            if voice_choice not in VOICE_SAMPLES:
                 return None, "⚠️ Vui lòng chọn một giọng mẫu."
            ref_audio_path = VOICE_SAMPLES[voice_choice]["audio"]
            ref_text_path = VOICE_SAMPLES[voice_choice]["text"]
            
            if not os.path.exists(ref_audio_path):
                 return None, f"❌ Không tìm thấy file audio: {ref_audio_path}"
                 
            with open(ref_text_path, "r", encoding="utf-8") as f:
                ref_text_raw = f.read()
            print(f"🎤 Mode: Preset Voice ({voice_choice})")

        # Inference & Đo thời gian
        print(f"📝 Text: {text[:50]}...")
        
        start_time = time.time() # <--- Bắt đầu bấm giờ
        
        ref_codes = tts.encode_reference(ref_audio_path)
        wav = tts.infer(text, ref_codes, ref_text_raw)
        
        end_time = time.time()   # <--- Kết thúc bấm giờ
        process_time = end_time - start_time # <--- Tính thời gian xử lý
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, wav, 24000)
            output_path = tmp_file.name
        
        # <--- Cập nhật thông báo kết quả
        return output_path, f"✅ Thành công! (Mất {process_time:.2f} giây để tạo)"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Lỗi hệ thống: {str(e)}"

# --- 4. UI SETUP ---
theme = gr.themes.Ocean(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui'],
).set(
    button_primary_background_fill="linear-gradient(90deg, #6366f1 0%, #0ea5e9 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4f46e5 0%, #0284c7 100%)",
    block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
)

# <--- CSS ĐÃ SỬA (Background xanh đen + Chữ sáng)
css = """
.container { max-width: 1200px; margin: auto; }
.header-box { 
    text-align: center; 
    margin-bottom: 25px; 
    padding: 25px; 
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); /* Xanh đen (Slate 900 -> 800) */
    border-radius: 12px; 
    border: 1px solid #334155; 
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
}
.header-title { 
    font-size: 2.5rem; 
    font-weight: 800; 
    color: white; /* Chữ trắng */
    background: -webkit-linear-gradient(45deg, #60A5FA, #22D3EE); /* Gradient xanh sáng cho chữ nổi bật */
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    margin-bottom: 10px; 
}
.header-desc {
    font-size: 1.1rem; 
    color: #cbd5e1; /* Màu xám sáng (Slate-300) */
    margin-bottom: 15px;
}
.link-group a { 
    text-decoration: none; 
    margin: 0 10px; 
    font-weight: 600; 
    color: #94a3b8; /* Màu link sáng hơn chút */
    transition: color 0.2s; 
}
.link-group a:hover { color: #38bdf8; text-shadow: 0 0 5px rgba(56, 189, 248, 0.5); }

.status-box { font-weight: bold; text-align: center; border: none; background: transparent; }
.warning-note { 
    background-color: #fff7ed; 
    border-left: 4px solid #f97316; 
    padding: 12px; 
    color: #9a3412; 
    font-size: 0.9rem; 
    border-radius: 4px;
    margin-top: 10px;
    margin-bottom: 10px;
}
"""

EXAMPLES_LIST = [
    # Nam Miền Nam
    ["Về miền Tây không chỉ để ngắm nhìn sông nước hữu tình, mà còn để cảm nhận tấm chân tình của người dân nơi đây. Cùng ngồi xuồng ba lá len lỏi qua rặng dừa nước, nghe câu vọng cổ ngọt ngào thì còn gì bằng.", "Vĩnh (nam miền Nam)"],
    
    # Nam Miền Bắc
    ["Hà Nội những ngày vào thu mang một vẻ đẹp trầm mặc và cổ kính đến lạ thường. Đi dạo quanh Hồ Gươm vào sáng sớm, hít hà mùi hoa sữa nồng nàn và thưởng thức chút cốm làng Vòng là trải nghiệm khó quên.", "Bình (nam miền Bắc)"],
    
    # Nam Miền Bắc
    ["Sự bùng nổ của trí tuệ nhân tạo đang định hình lại cách chúng ta làm việc và sinh sống. Từ xe tự lái đến trợ lý ảo thông minh, công nghệ đang dần xóa nhòa ranh giới giữa thực tại và những bộ phim viễn tưởng.", "Tuyên (nam miền Bắc)"],
    
    # Nam Miền Nam
    ["Sài Gòn hối hả là thế, nhưng chỉ cần tấp vào một quán cà phê ven đường, gọi ly bạc xỉu đá và ngắm nhìn dòng người qua lại, bạn sẽ thấy thành phố này cũng có những khoảng lặng thật bình yên và đáng yêu.", "Nguyên (nam miền Nam)"],
    
    # Nam Miền Nam
    ["Để đảm bảo tiến độ dự án quan trọng này, chúng ta cần tập trung tối đa nguồn lực và phối hợp chặt chẽ giữa các phòng ban. Mọi khó khăn phát sinh cần được báo cáo ngay lập tức để ban lãnh đạo xử lý kịp thời.", "Sơn (nam miền Nam)"],
    
    # Nữ Miền Nam
    ["Dạ em chào anh chị, hiện tại bên em đang có chương trình ưu đãi đặc biệt cho căn hộ hướng sông này. Với thiết kế hiện đại và không gian xanh mát, đây chắc chắn là tổ ấm lý tưởng mà gia đình mình đang tìm kiếm.", "Đoan (nữ miền Nam)"],
    
    # Nữ Miền Bắc
    ["Dưới cơn mưa phùn lất phất của những ngày cuối đông, em khẽ nép vào vai anh, cảm nhận hơi ấm lan tỏa. Những khoảnh khắc bình dị như thế này khiến em nhận ra rằng, hạnh phúc đôi khi chỉ đơn giản là được ở bên nhau.", "Ngọc (nữ miền Bắc)"],

    # Nữ Miền Bắc
    ["Thay mặt phi hành đoàn, xin chào mừng quý khách đến với chuyến bay vi en 2024. Quý khách vui lòng thắt dây an toàn, dựng thẳng lưng ghế và gập bàn ăn phía trước để chuẩn bị cho máy bay cất cánh trong ít phút nữa.", "Hương (nữ miền Bắc)"],
    
    # Nữ Miền Bắc
    ["Ngày xửa ngày xưa, ở một ngôi làng nọ có cô Tấm xinh đẹp, nết na nhưng sớm mồ côi mẹ. Dù bị mẹ kế và Cám hãm hại đủ đường, Tấm vẫn giữ được tấm lòng lương thiện và cuối cùng tìm được hạnh phúc xứng đáng.", "Ly (nữ miền Bắc)"],
]

with gr.Blocks(theme=theme, css=css, title="VieNeu-TTS Studio") as demo:
    
    with gr.Column(elem_classes="container"):
        # Header - Cập nhật class cho HTML
        gr.HTML("""
            <div class="header-box">
                <div class="header-title">🎙️ VieNeu-TTS Studio</div>
                <div class="header-desc">
                    Phiên bản: VieNeu-TTS-1000h (model mới nhất, train trên 1000 giờ dữ liệu)
                </div>
                <div class="link-group">
                    <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS" target="_blank">🤗 Model Card</a> • 
                    <a href="https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h" target="_blank">📖 Dataset 1000h</a> • 
                    <a href="https://github.com/pnnbao97/VieNeu-TTS" target="_blank">🦜 GitHub</a>
                </div>
            </div>
        """)
    
    with gr.Row(elem_classes="container", equal_height=False):
        
        # --- LEFT: INPUT ---
        with gr.Column(scale=3, variant="panel"):
            gr.Markdown("### 📝 Văn bản đầu vào")
            text_input = gr.Textbox(
                label="Nhập văn bản",
                placeholder="Nhập nội dung tiếng Việt cần chuyển thành giọng nói...",
                lines=4,
                value="Hà Nội những ngày vào thu mang một vẻ đẹp trầm mặc và cổ kính đến lạ thường. Đi dạo quanh Hồ Gươm vào sáng sớm, hít hà mùi hoa sữa nồng nàn và thưởng thức chút cốm làng Vòng là trải nghiệm khó quên.",
                show_label=False
            )
            
            # Counter + Warning
            with gr.Row():
                char_count = gr.HTML("<div style='text-align: right; color: #64748B; font-size: 0.8rem;'>0 / 250 ký tự</div>")
            
            gr.Markdown("### 🗣️ Chọn giọng đọc")
            with gr.Tabs() as tabs:
                with gr.TabItem("👤 Giọng có sẵn (Preset)", id="preset_mode"):
                    voice_select = gr.Dropdown(
                        choices=list(VOICE_SAMPLES.keys()),
                        value="Bình (nam miền Bắc)",
                        label="Danh sách giọng",
                        interactive=True
                    )
                    with gr.Accordion("Thông tin giọng mẫu", open=False):
                        ref_audio_preview = gr.Audio(label="Audio mẫu", interactive=False, type="filepath")
                        ref_text_preview = gr.Markdown("...")

                with gr.TabItem("🎙️ Giọng tùy chỉnh (Custom)", id="custom_mode"):
                    gr.Markdown("Tải lên giọng của bạn (Zero-shot Cloning)")
                    custom_audio = gr.Audio(label="File ghi âm (.wav)", type="filepath")
                    custom_text = gr.Textbox(label="Nội dung ghi âm", placeholder="Nhập chính xác lời thoại...")

            current_mode = gr.Textbox(visible=False, value="preset_mode")
            btn_generate = gr.Button("Tổng hợp giọng nói", variant="primary", size="lg")

        # --- RIGHT: OUTPUT ---
        with gr.Column(scale=2):
            gr.Markdown("### 🎧 Kết quả")
            with gr.Group():
                audio_output = gr.Audio(label="Audio đầu ra", type="filepath", show_download_button=True, autoplay=True)
                status_output = gr.Textbox(label="Trạng thái", show_label=False, elem_classes="status-box", placeholder="Sẵn sàng...")

    # --- EXAMPLES ---
    with gr.Row(elem_classes="container"):
        with gr.Column():
            gr.Markdown("### 📚 Ví dụ mẫu")
            gr.Examples(examples=EXAMPLES_LIST, inputs=[text_input, voice_select], label="Thử nghiệm nhanh")

    # --- LOGIC ---
    def update_count(text):
        l = len(text)
        if l > 250:
            color = "#dc2626" # Red
            msg = f"⚠️ <b>{l} / 250</b> - Quá giới hạn!"
        elif l > 200:
            color = "#ea580c" # Orange
            msg = f"{l} / 250"
        else:
            color = "#64748B" # Gray
            msg = f"{l} / 250 ký tự"
        return f"<div style='text-align: right; color: {color}; font-size: 0.8rem; font-weight: bold'>{msg}</div>"

    text_input.change(update_count, text_input, char_count)

    def update_ref_preview(voice):
        audio, text = load_reference_info(voice)
        return audio, f"> *\"{text}\"*"
    
    voice_select.change(update_ref_preview, voice_select, [ref_audio_preview, ref_text_preview])
    demo.load(update_ref_preview, voice_select, [ref_audio_preview, ref_text_preview])

    # Tab handling - FIXED WITH *ARGS
    tab_preset = tabs.children[0]
    tab_custom = tabs.children[1]
    
    # Dùng *args để nhận bất kỳ số lượng tham số nào (0 hoặc 1), tránh lỗi Warning
    tab_preset.select(fn=lambda *args: "preset_mode", inputs=None, outputs=current_mode)
    tab_custom.select(fn=lambda *args: "custom_mode", inputs=None, outputs=current_mode)

    btn_generate.click(
        fn=synthesize_speech,
        inputs=[text_input, voice_select, custom_audio, custom_text, current_mode],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        share=False
    )