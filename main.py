from vieneu_tts import VieNeuTTS
import soundfile as sf
import torch
import os

device = "cpu"

input_texts = [
    "Các khóa học trực tuyến đang giúp học sinh tiếp cận kiến thức mọi lúc mọi nơi. Giáo viên sử dụng video, bài tập tương tác và thảo luận trực tuyến để nâng cao hiệu quả học tập.",

    "Các nghiên cứu về bệnh Alzheimer cho thấy tác dụng tích cực của các bài tập trí não và chế độ dinh dưỡng lành mạnh, giúp giảm tốc độ suy giảm trí nhớ ở người cao tuổi.",

    "Một tiểu thuyết trinh thám hiện đại dẫn dắt độc giả qua những tình tiết phức tạp, bí ẩn, kết hợp yếu tố tâm lý sâu sắc khiến người đọc luôn hồi hộp theo dõi diễn biến câu chuyện.",

    "Các nhà khoa học nghiên cứu gen người phát hiện những đột biến mới liên quan đến bệnh di truyền. Điều này giúp nâng cao khả năng chẩn đoán và điều trị.",
]

output_dir = "./output_audio"
os.makedirs(output_dir, exist_ok=True)

def main(backbone="pnnbao-ump/VieNeu-TTS-q4-gguf", codec="neuphonic/neucodec-onnx-decoder"):
    """
    In the sample directory, there are wav files and txt files with matching names.
    These are pre-prepared reference files for testing with Vietnamese names:
    - Bình (nam miền Bắc) - Male, North accent
    - Tuyên (nam miền Bắc) - Male, North accent
    - Nguyên (nam miền Nam) - Male, South accent
    - Sơn (nam miền Nam) - Male, South accent
    - Vĩnh (nam miền Nam) - Male, South accent
    - Hương (nữ miền Bắc) - Female, North accent
    - Ly (nữ miền Bắc) - Female, North accent
    - Ngọc (nữ miền Bắc) - Female, North accent
    - Đoan (nữ miền Nam) - Female, South accent
    - Dung (nữ miền Nam) - Female, South accent
    
    Note: The model can clone any voice you provide (with corresponding text).
    However, quality may not match the sample files. For best results, finetune
    the model on your target voice. See finetune guide at:
    https://github.com/pnnbao-ump/VieNeuTTS/blob/main/finetune.ipynb
    """
    # Male voice (South accent)
    ref_audio_path = "./sample/Vĩnh (nam miền Nam).wav"
    ref_text_path = "./sample/Vĩnh (nam miền Nam).txt"
    ref_codes_path = "./sample/Vĩnh (nam miền Nam).pt"
    
    # Female voice (South accent) - uncomment to use
    # ref_audio_path = "./sample/Đoan (nữ miền Nam).wav"
    # ref_text_path = "./sample/Đoan (nữ miền Nam).txt"

    ref_text_raw = open(ref_text_path, "r", encoding="utf-8").read()
    
    if not ref_audio_path or not ref_text_raw:
        print("No reference audio or text provided.")
        return None

    # Initialize VieNeuTTS-1000h
    tts = VieNeuTTS(
        backbone_repo=backbone,
        backbone_device=device,
        codec_repo=codec,
        codec_device=device
    )

    if codec == "neuphonic/neucodec-onnx-decoder":
        print("Load reference codes...")
        ref_codes = torch.load(ref_codes_path, map_location=device)
    else:
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
