from vieneu_tts import VieNeuTTS
import soundfile as sf
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

input_texts = [
    "Các khóa học trực tuyến đang giúp học sinh tiếp cận kiến thức mọi lúc mọi nơi. Giáo viên sử dụng video, bài tập tương tác và thảo luận trực tuyến để nâng cao hiệu quả học tập.",

    "Các nghiên cứu về bệnh Alzheimer cho thấy tác dụng tích cực của các bài tập trí não và chế độ dinh dưỡng lành mạnh, giúp giảm tốc độ suy giảm trí nhớ ở người cao tuổi.",

    "Một tiểu thuyết trinh thám hiện đại dẫn dắt độc giả qua những tình tiết phức tạp, bí ẩn, kết hợp yếu tố tâm lý sâu sắc khiến người đọc luôn hồi hộp theo dõi diễn biến câu chuyện.",

    "Các nhà khoa học nghiên cứu gen người phát hiện những đột biến mới liên quan đến bệnh di truyền. Điều này giúp nâng cao khả năng chẩn đoán và điều trị.",
]

output_dir = "./output_audio"
os.makedirs(output_dir, exist_ok=True)

def main(backbone="pnnbao-ump/VieNeu-TTS", codec="neuphonic/neucodec"):
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
        backbone_device=device,
        codec_repo=codec,
        codec_device=device
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
