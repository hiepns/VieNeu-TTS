import gradio as gr
import soundfile as sf
import tempfile
import torch
from vieneu_tts import VieNeuTTS
import os
import time
import numpy as np
import re
from typing import Generator
import queue
import threading
import yaml
from utils.core_utils import split_text_into_chunks

print("⏳ Đang khởi động VieNeu-TTS...")

# --- CONSTANTS & CONFIG ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}
except Exception as e:
    raise RuntimeError(f"Không thể đọc config.yaml: {e}")

BACKBONE_CONFIGS = _config.get("backbone_configs", {})
CODEC_CONFIGS = _config.get("codec_configs", {})
VOICE_SAMPLES = _config.get("voice_samples", {})

_text_settings = _config.get("text_settings", {})
MAX_CHARS_PER_CHUNK = _text_settings.get("max_chars_per_chunk", 256)
MAX_TOTAL_CHARS_STREAMING = _text_settings.get("max_total_chars_streaming", 3000)

if not BACKBONE_CONFIGS or not CODEC_CONFIGS:
    raise ValueError("config.yaml thiếu backbone_configs hoặc codec_configs")
if not VOICE_SAMPLES:
    raise ValueError("config.yaml thiếu voice_samples")

# --- 1. MODEL CONFIGURATION ---
# Global model instance
tts = None
current_backbone = None
current_codec = None
model_loaded = False  # ✨ THÊM STATE

def load_model(backbone_choice, codec_choice, device_choice):
    """Load model with specified configuration"""
    global tts, current_backbone, current_codec, model_loaded
    
    # ✨ Trả về nhiều outputs để update UI ngay lập tức
    yield (
        "⏳ Đang tải model, vui lòng đợi...",
        gr.update(interactive=False),  # Disable nút "Bắt đầu"
        gr.update(interactive=False)   # Disable nút "Tải Model"
    )
    
    try:
        backbone_config = BACKBONE_CONFIGS[backbone_choice]
        codec_config = CODEC_CONFIGS[codec_choice]
        
        # Determine devices
        if device_choice == "Auto":
            if "GGUF" in backbone_choice:
                backbone_device = "gpu" if torch.cuda.is_available() else "cpu"
            else:
                backbone_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if "ONNX" in codec_choice:
                codec_device = "cpu"
            else:
                codec_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            backbone_device = device_choice.lower()
            codec_device = device_choice.lower()
            
            if "ONNX" in codec_choice:
                codec_device = "cpu"
        
        if "GGUF" in backbone_choice and backbone_device == "cuda":
            backbone_device = "gpu"
        
        print(f"📦 Đang tải model...")
        print(f"   Backbone: {backbone_config['repo']} on {backbone_device}")
        print(f"   Codec: {codec_config['repo']} on {codec_device}")
        
        tts = VieNeuTTS(
            backbone_repo=backbone_config["repo"],
            backbone_device=backbone_device,
            codec_repo=codec_config["repo"],
            codec_device=codec_device
        )
        
        current_backbone = backbone_choice
        current_codec = codec_choice
        model_loaded = True  # ✨ Đánh dấu đã load xong
        
        streaming_support = "✅ Có" if backbone_config['supports_streaming'] else "❌ Không"
        preencoded_note = "\n⚠️ Codec này cần sử dụng pre-encoded codes (.pt files)" if codec_config['use_preencoded'] else ""
        
        success_msg = (
            f"✅ Model đã tải thành công!\n\n"
            f"🦜 Model Device: {backbone_device.upper()}\n\n"
            f"🎵 Codec Device: {codec_device.upper()}{preencoded_note}"
        )
        
        yield (
            success_msg,
            gr.update(interactive=True),   # ✨ Enable nút "Bắt đầu"
            gr.update(interactive=True)    # Enable nút "Tải Model"
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        model_loaded = False
        
        yield (
            f"❌ Lỗi khi tải model: {str(e)}",
            gr.update(interactive=False),  # Vẫn disable nút "Bắt đầu"
            gr.update(interactive=True)    # Enable nút "Tải Model" để thử lại
        )

# --- 2. DATA & HELPERS ---
GGUF_ALLOWED_VOICES = [
    "Vĩnh (nam miền Nam)",
    "Bình (nam miền Bắc)",
    "Ngọc (nữ miền Bắc)",
    "Dung (nữ miền Nam)",
]

def get_voice_options(backbone_choice: str):
    """Filter voice options: GGUF only shows the 4 allowed voices."""
    if "GGUF" in backbone_choice:
        return [v for v in GGUF_ALLOWED_VOICES if v in VOICE_SAMPLES]
    return list(VOICE_SAMPLES.keys())

def update_voice_dropdown(backbone_choice: str, current_voice: str):
    options = get_voice_options(backbone_choice)
    new_value = current_voice if current_voice in options else (options[0] if options else None)
    return gr.update(choices=options, value=new_value)

# --- 3. CORE LOGIC FUNCTIONS ---
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

def synthesize_speech(text, voice_choice, custom_audio, custom_text, mode_tab, generation_mode):
    """Synthesis with model check"""
    global tts, current_backbone, current_codec, model_loaded
    
    # ✨ Kiểm tra model đã load chưa
    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return
    
    if not text or text.strip() == "":
        yield None, "⚠️ Vui lòng nhập văn bản!"
        return
    
    raw_text = text.strip()
    
    codec_config = CODEC_CONFIGS[current_codec]
    use_preencoded = codec_config['use_preencoded']
    
    # Setup Reference
    if mode_tab == "custom_mode":
        if custom_audio is None or not custom_text:
            yield None, "⚠️ Thiếu Audio hoặc Text mẫu custom."
            return
        ref_audio_path = custom_audio
        ref_text_raw = custom_text
        ref_codes_path = None
    else:
        if voice_choice not in VOICE_SAMPLES:
            yield None, "⚠️ Vui lòng chọn giọng mẫu."
            return
        ref_audio_path = VOICE_SAMPLES[voice_choice]["audio"]
        ref_text_path = VOICE_SAMPLES[voice_choice]["text"]
        ref_codes_path = VOICE_SAMPLES[voice_choice]["codes"]
        
        if not os.path.exists(ref_audio_path):
            yield None, "❌ Không tìm thấy file audio mẫu."
            return
        
        with open(ref_text_path, "r", encoding="utf-8") as f:
            ref_text_raw = f.read()
    
    yield None, "📄 Đang xử lý Reference..."
    
    # Encode reference
    try:
        if use_preencoded and ref_codes_path and os.path.exists(ref_codes_path):
            ref_codes = torch.load(ref_codes_path, map_location="cpu")
        else:
            ref_codes = tts.encode_reference(ref_audio_path)
        
        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()
    except Exception as e:
        yield None, f"❌ Lỗi xử lý reference: {e}"
        return
    
    text_chunks = split_text_into_chunks(raw_text, max_chars=MAX_CHARS_PER_CHUNK)
    total_chunks = len(text_chunks)
    
    # === STANDARD MODE ===
    if generation_mode == "Standard (Một lần)":
        yield None, f"🚀 Bắt đầu tổng hợp chế độ Standard ({total_chunks} đoạn)..."
        
        all_audio_segments = []
        sr = 24000
        silence_pad = np.zeros(int(sr * 0.15), dtype=np.float32)
        
        start_time = time.time()
        
        try:
            for i, chunk in enumerate(text_chunks):
                yield None, f"⏳ Đang xử lý đoạn {i+1}/{total_chunks}..."
                
                chunk_wav = tts.infer(chunk, ref_codes, ref_text_raw)
                
                if chunk_wav is not None and len(chunk_wav) > 0:
                    all_audio_segments.append(chunk_wav)
                    if i < total_chunks - 1:
                        all_audio_segments.append(silence_pad)
            
            if not all_audio_segments:
                yield None, "❌ Không sinh được audio nào."
                return
            
            yield None, "💾 Đang ghép file và lưu..."
            
            final_wav = np.concatenate(all_audio_segments)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                output_path = tmp.name
            
            process_time = time.time() - start_time
            yield output_path, f"✅ Hoàn tất! (Tổng thời gian: {process_time:.2f}s)"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield None, f"❌ Lỗi Standard Mode: {str(e)}"
            return
    
    # === STREAMING MODE ===
    else:
        sr = 24000
        crossfade_samples = int(sr * 0.03)
        audio_queue = queue.Queue(maxsize=100)
        PRE_BUFFER_SIZE = 3
        
        end_event = threading.Event()
        error_event = threading.Event()
        error_msg = ""
        
        def producer_thread():
            nonlocal error_msg
            try:
                previous_tail = None
                chunk_count = 0
                
                for i, chunk_text in enumerate(text_chunks):
                    stream_gen = tts.infer_stream(chunk_text, ref_codes, ref_text_raw)
                    
                    for part_idx, audio_part in enumerate(stream_gen):
                        if audio_part is None or len(audio_part) == 0:
                            continue
                        
                        if previous_tail is not None and len(previous_tail) > 0:
                            overlap = min(len(previous_tail), len(audio_part), crossfade_samples)
                            if overlap > 0:
                                fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                                
                                blended = (audio_part[:overlap] * fade_in + 
                                         previous_tail[-overlap:] * fade_out)
                                
                                processed = np.concatenate([
                                    previous_tail[:-overlap] if len(previous_tail) > overlap else np.array([]),
                                    blended,
                                    audio_part[overlap:]
                                ])
                            else:
                                processed = np.concatenate([previous_tail, audio_part])
                            
                            tail_size = min(crossfade_samples, len(processed))
                            previous_tail = processed[-tail_size:].copy()
                            output_chunk = processed[:-tail_size] if len(processed) > tail_size else processed
                        else:
                            tail_size = min(crossfade_samples, len(audio_part))
                            previous_tail = audio_part[-tail_size:].copy()
                            output_chunk = audio_part[:-tail_size] if len(audio_part) > tail_size else audio_part
                        
                        if len(output_chunk) > 0:
                            audio_queue.put((sr, output_chunk))
                            chunk_count += 1
                
                if previous_tail is not None and len(previous_tail) > 0:
                    audio_queue.put((sr, previous_tail))
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                error_event.set()
            finally:
                end_event.set()
                audio_queue.put(None)
        
        threading.Thread(target=producer_thread, daemon=True).start()
        
        yield (sr, np.zeros(int(sr * 0.05))), "🔄 Đang buffering..."
        
        pre_buffer = []
        while len(pre_buffer) < PRE_BUFFER_SIZE:
            try:
                item = audio_queue.get(timeout=5.0)
                if item is None:
                    break
                pre_buffer.append(item)
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    return
                break
        
        full_audio_buffer = []
        for sr, audio_data in pre_buffer:
            full_audio_buffer.append(audio_data)
            yield (sr, audio_data), "🔊 Đang phát..."
        
        while True:
            try:
                item = audio_queue.get(timeout=0.05)
                if item is None:
                    break
                sr, audio_data = item
                full_audio_buffer.append(audio_data)
                yield (sr, audio_data), "🔊 Đang phát..."
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    break
                if end_event.is_set() and audio_queue.empty():
                    break
                continue
        
        if full_audio_buffer:
            final_wav = np.concatenate(full_audio_buffer)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                yield tmp.name, "✅ Hoàn tất Streaming!"

# --- 4. UI SETUP ---
theme = gr.themes.Ocean(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui'],
).set(
    button_primary_background_fill="linear-gradient(90deg, #6366f1 0%, #0ea5e9 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4f46e5 0%, #0284c7 100%)",
)

css = """
.container { max-width: 1400px; margin: auto; }
.header-box {
    text-align: center;
    margin-bottom: 25px;
    padding: 25px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px;
    color: white;
}
.header-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, #60A5FA, #22D3EE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.status-box {
    font-weight: bold;
    text-align: center;
    border: none;
    background: transparent;
}
"""

EXAMPLES_LIST = [
    ["Về miền Tây không chỉ để ngắm nhìn sông nước hữu tình, mà còn để cảm nhận tấm chân tình của người dân nơi đây.", "Vĩnh (nam miền Nam)"],
    ["Hà Nội những ngày vào thu mang một vẻ đẹp trầm mặc và cổ kính đến lạ thường.", "Bình (nam miền Bắc)"],
]

with gr.Blocks(theme=theme, css=css, title="VieNeu-TTS") as demo:
    with gr.Column(elem_classes="container"):
        gr.HTML("""
<div class="header-box">
    <h1 class="header-title">🦜 VieNeu-TTS Studio</h1>
</div>
        """)
        
        # --- CONFIGURATION ---
        with gr.Group():
            with gr.Row():
                backbone_select = gr.Dropdown(list(BACKBONE_CONFIGS.keys()), value="GGUF Q8", label="🦜 Backbone")
                codec_select = gr.Dropdown(list(CODEC_CONFIGS.keys()), value="NeuCodec (Standard)", label="🎵 Codec")
                device_choice = gr.Radio(["Auto", "CPU", "CUDA"], value="Auto", label="🖥️ Device")
            
            
            btn_load = gr.Button("🔄 Tải Model", variant="primary")
            model_status = gr.Markdown("⏳ Chưa tải model.")
        
        with gr.Row(elem_classes="container"):
            # --- INPUT ---
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label=f"Văn bản (Streaming hỗ trợ tới {MAX_TOTAL_CHARS_STREAMING} ký tự, chia chunk {MAX_CHARS_PER_CHUNK} ký tự)",
                    lines=4,
                    value="Hà Nội, trái tim của Việt Nam, là một thành phố ngàn năm văn hiến với bề dày lịch sử và văn hóa độc đáo. Bước chân trên những con phố cổ kính quanh Hồ Hoàn Kiếm, du khách như được du hành ngược thời gian, chiêm ngưỡng kiến trúc Pháp cổ điển hòa quyện với nét kiến trúc truyền thống Việt Nam. Mỗi con phố trong khu phố cổ mang một tên gọi đặc trưng, phản ánh nghề thủ công truyền thống từng thịnh hành nơi đây như phố Hàng Bạc, Hàng Đào, Hàng Mã. Ẩm thực Hà Nội cũng là một điểm nhấn đặc biệt, từ tô phở nóng hổi buổi sáng, bún chả thơm lừng trưa hè, đến chè Thái ngọt ngào chiều thu. Những món ăn dân dã này đã trở thành biểu tượng của văn hóa ẩm thực Việt, được cả thế giới yêu mến. Người Hà Nội nổi tiếng với tính cách hiền hòa, lịch thiệp nhưng cũng rất cầu toàn trong từng chi tiết nhỏ, từ cách pha trà sen cho đến cách chọn hoa sen tây để thưởng trà.",
                )
                
                with gr.Tabs() as tabs:
                    with gr.TabItem("👤 Preset", id="preset_mode"):
                        initial_voices = get_voice_options("GGUF Q8")
                        default_voice = initial_voices[0] if initial_voices else None
                        voice_select = gr.Dropdown(initial_voices, value=default_voice, label="Giọng mẫu")
                    
                    with gr.TabItem("🎙️ Custom", id="custom_mode"):
                        custom_audio = gr.Audio(label="File mẫu (.wav)", type="filepath")
                        custom_text = gr.Textbox(label="Lời thoại mẫu")
                
                generation_mode = gr.Radio(
                    ["Standard (Một lần)"],
                    value="Standard (Một lần)",
                    label="Chế độ sinh"
                )
                
                current_mode = gr.Textbox(visible=False, value="preset_mode")
                
                # ✨ NÚT BẮT ĐẦU - MẶC ĐỊNH DISABLE
                btn_generate = gr.Button("🎵 Bắt đầu", variant="primary", size="lg", interactive=False)
            
            # --- OUTPUT ---
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Kết quả",
                    type="filepath",
                    autoplay=True,
                    show_download_button=True
                )
                status_output = gr.Textbox(label="Trạng thái", elem_classes="status-box")
        
        # --- EVENT HANDLERS ---
        def update_info(backbone):
            return f"Streaming: {'✅' if BACKBONE_CONFIGS[backbone]['supports_streaming'] else '❌'}"
        
        backbone_select.change(update_info, backbone_select, model_status)
        backbone_select.change(update_voice_dropdown, [backbone_select, voice_select], voice_select)
        
        tabs.children[0].select(lambda: "preset_mode", outputs=current_mode)
        tabs.children[1].select(lambda: "custom_mode", outputs=current_mode)
        
        # ✨ CẬP NHẬT EVENT HANDLER CHO NÚT LOAD
        btn_load.click(
            fn=load_model,
            inputs=[backbone_select, codec_select, device_choice],
            outputs=[model_status, btn_generate, btn_load]  # Update cả 3 components
        )
        
        btn_generate.click(
            fn=synthesize_speech,
            inputs=[text_input, voice_select, custom_audio, custom_text, current_mode, generation_mode],
            outputs=[audio_output, status_output]
        )

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)