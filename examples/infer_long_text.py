import argparse
import os
import re
import sys
from pathlib import Path
from typing import List
import numpy as np
import soundfile as sf
import torch
from vieneu_tts import VieNeuTTS


def split_text_into_chunks(text: str, max_chars: int = 256) -> List[str]:
    """
    Split raw text into chunks no longer than max_chars.
    Preference is given to sentence boundaries; otherwise falls back to word-based splitting.
    """
    sentences = re.split(r"(?<=[\.\!\?\…])\s+", text.strip())
    chunks: List[str] = []
    buffer = ""

    def flush_buffer():
        nonlocal buffer
        if buffer:
            chunks.append(buffer.strip())
            buffer = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If single sentence already fits, try to append to current buffer
        if len(sentence) <= max_chars:
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                flush_buffer()
                buffer = sentence
            continue

        # Fallback: sentence too long, break by words
        flush_buffer()
        words = sentence.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) > max_chars and current:
                chunks.append(current.strip())
                current = word
            else:
                current = candidate
        if current:
            chunks.append(current.strip())

    flush_buffer()
    return [chunk for chunk in chunks if chunk]


def infer_long_text(
    text: str,
    ref_audio_path: str,
    ref_text_path: str,
    output_path: str,
    chunk_dir: str | None = None,
    max_chars: int = 256,
    backbone_repo: str = "pnnbao-ump/VieNeu-TTS",
    codec_repo: str = "neuphonic/neucodec",
    device: str | None = None,
) -> str:
    """
    Generate speech for long-form text by chunking into manageable segments.

    Returns:
        The path to the combined audio file.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device not in {"cuda", "cpu"}:
        raise ValueError("Device must be either 'cuda' or 'cpu'.")

    raw_text = text.strip()
    if not raw_text:
        raise ValueError("Input text is empty.")

    chunks = split_text_into_chunks(raw_text, max_chars=max_chars)
    if not chunks:
        raise ValueError("Text could not be segmented into valid chunks.")

    print(f"📄 Total chunks: {len(chunks)} (≤ {max_chars} chars each)")

    if chunk_dir:
        os.makedirs(chunk_dir, exist_ok=True)

    ref_text_raw = Path(ref_text_path).read_text(encoding="utf-8")

    tts = VieNeuTTS(
        backbone_repo=backbone_repo,
        backbone_device=device,
        codec_repo=codec_repo,
        codec_device=device,
    )

    print("🎧 Encoding reference audio...")
    ref_codes = tts.encode_reference(ref_audio_path)

    generated_segments: List[np.ndarray] = []

    for idx, chunk in enumerate(chunks, start=1):
        print(f"🎙️ Chunk {idx}/{len(chunks)} | {len(chunk)} chars")
        wav = tts.infer(chunk, ref_codes, ref_text_raw)
        generated_segments.append(wav)

        if chunk_dir:
            chunk_path = os.path.join(chunk_dir, f"chunk_{idx:03d}.wav")
            sf.write(chunk_path, wav, 24_000)

    combined_audio = np.concatenate(generated_segments)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, combined_audio, 24_000)

    print(f"✅ Saved combined audio to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer long text with VieNeu-TTS")
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text",
        help="Raw UTF-8 text content to synthesize.",
    )
    text_group.add_argument(
        "--text-file",
        help="Path to a UTF-8 text file to synthesize.",
    )
    parser.add_argument(
        "--ref-audio",
        default="./sample/Vĩnh (nam miền Nam).wav",
        help="Path to reference audio (.wav). Default: ./sample/Vĩnh (nam miền Nam).wav"
    )
    parser.add_argument(
        "--ref-text",
        default="./sample/Vĩnh (nam miền Nam).txt",
        help="Path to reference text (UTF-8). Default: ./sample/Vĩnh (nam miền Nam).txt"
    )
    parser.add_argument(
        "--output",
        default="./output_audio/long_text.wav",
        help="Path to save the combined audio output.",
    )
    parser.add_argument(
        "--chunk-output-dir",
        default=None,
        help="Optional directory to save individual chunk audio files.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=256,
        help="Maximum characters per chunk before TTS inference.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to run inference on (auto=CUDA if available).",
    )
    parser.add_argument(
        "--backbone",
        default="pnnbao-ump/VieNeu-TTS",
        help="Backbone repository ID or local path.",
    )
    parser.add_argument(
        "--codec",
        default="neuphonic/neucodec",
        help="Codec repository ID or local path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ref_audio_path = Path(args.ref_audio)
    if not ref_audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    ref_text_path = Path(args.ref_text)
    if not ref_text_path.exists():
        raise FileNotFoundError(f"Reference text not found: {ref_text_path}")

    if args.text_file:
        text_path = Path(args.text_file)
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found: {text_path}")
        raw_text = text_path.read_text(encoding="utf-8")
    else:
        raw_text = args.text.strip()
        if not raw_text:
            raise ValueError("Provided text is empty.")
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    infer_long_text(
        text=raw_text,
        ref_audio_path=str(ref_audio_path),
        ref_text_path=str(ref_text_path),
        output_path=args.output,
        chunk_dir=args.chunk_output_dir,
        max_chars=args.max_chars,
        backbone_repo=args.backbone,
        codec_repo=args.codec,
        device=device,
    )


if __name__ == "__main__":
    main()

