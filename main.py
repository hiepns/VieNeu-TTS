from vieneutts import VieNeuTTS
import soundfile as sf
import os

# 10 ví dụ để test TTS
input_texts = [
    "Các khóa học trực tuyến đang giúp học sinh tiếp cận kiến thức mọi lúc mọi nơi. Giáo viên sử dụng video, bài tập tương tác và thảo luận trực tuyến để nâng cao hiệu quả học tập.",

    "Các nghiên cứu về bệnh Alzheimer cho thấy tác dụng tích cực của các bài tập trí não và chế độ dinh dưỡng lành mạnh, giúp giảm tốc độ suy giảm trí nhớ ở người cao tuổi.",

    "Một tiểu thuyết trinh thám hiện đại dẫn dắt độc giả qua những tình tiết phức tạp, bí ẩn, kết hợp yếu tố tâm lý sâu sắc khiến người đọc luôn hồi hộp theo dõi diễn biến câu chuyện.",

    "Các nhà khoa học nghiên cứu gen người phát hiện những đột biến mới liên quan đến bệnh di truyền. Điều này giúp nâng cao khả năng chẩn đoán và điều trị.",
]

output_dir = "./output_audio"
os.makedirs(output_dir, exist_ok=True)

def main(backbone="pnnbao-ump/VieNeu-TTS", codec="neuphonic/neucodec"):
    # Nam miền Nam
    ref_audio_path = "./sample/id_0001.wav"
    ref_text = "./sample/id_0001.txt"
    # Nữ miền Nam
    # ref_audio_path = "./sample/id_0002.wav"
    # ref_text = "./sample/id_0002.txt"
    ref_text = open(ref_text, "r", encoding="utf-8").read()
    if not ref_audio_path or not ref_text:
        print("No reference audio or text provided.")
        return None

    # Initialize VieNeuTTS with the desired model and codec
    tts = VieNeuTTS(
        backbone_repo=backbone,
        backbone_device="cuda",
        codec_repo=codec,
        codec_device="cuda"
    )

    print("Encoding reference audio")
    ref_codes = tts.encode_reference(ref_audio_path)

    # Loop through all input texts
    for i, text in enumerate(input_texts, 1):
        print(f"Generating audio for example {i}: {text}")
        wav = tts.infer(text, ref_codes, ref_text)
        output_path = os.path.join(output_dir, f"output_{i}.wav")
        sf.write(output_path, wav, 24000)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()