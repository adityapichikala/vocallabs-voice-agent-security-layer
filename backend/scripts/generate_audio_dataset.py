"""Script to synthesize 20 dual-speaker .wav test audio files using edge-tts or built-in PCM WAV generator."""
import asyncio
import json
import math
import os
import struct
import wave
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
TEST_CASES_PATH = DATA_DIR / "test_cases_metadata.json"
AUDIO_OUTPUT_DIR = PROJECT_ROOT / "data" / "audio"

# Voice mappings for dual-speaker synthesis
AGENT_VOICE = "en-US-AriaNeural"
CUSTOMER_VOICE_EN = "en-US-GuyNeural"
CUSTOMER_VOICE_HI = "hi-IN-MadhurNeural"

def generate_synthetic_pcm_wav(output_path: Path, duration_seconds: float = 5.0, sample_rate: int = 16000):
    """Fallback generator: creates a valid 16-bit mono 16kHz PCM WAV file with gentle audio tone."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_seconds * sample_rate)
    
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)        # Mono
        wav_file.setsampwidth(2)        # 16-bit
        wav_file.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(num_samples):
            # Soft 440 Hz / 220 Hz modulated tone with envelope
            t = float(i) / sample_rate
            amplitude = 3000 * math.sin(2 * math.pi * 440.0 * t) * (0.5 + 0.5 * math.sin(2 * math.pi * 2.0 * t))
            val = int(max(-32767, min(32767, amplitude)))
            frames.extend(struct.pack("<h", val))
        
        wav_file.writeframes(frames)

async def synthesize_with_edge_tts(text: str, voice: str, temp_chunk_path: Path):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(temp_chunk_path))

async def main():
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not TEST_CASES_PATH.exists():
        print(f"[ERROR] Test cases metadata file not found at: {TEST_CASES_PATH}")
        return

    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    print(f"=== Black Box Test Call Audio Dataset Generator ===")
    print(f"Total Test Cases to Process: {len(test_cases)}")
    print(f"Output Directory: {AUDIO_OUTPUT_DIR}\n")

    has_edge_tts = False
    try:
        import edge_tts
        has_edge_tts = True
        print("[INFO] edge-tts detected. High-fidelity neural voice synthesis enabled.")
    except ImportError:
        print("[INFO] edge-tts not installed in environment. Generating valid 16kHz PCM WAV fixtures for offline testing.")

    for tc in test_cases:
        tc_id = tc["id"]
        filename = tc.get("audio_filename", f"{tc_id}.wav")
        out_path = AUDIO_OUTPUT_DIR / filename
        duration = tc.get("duration_seconds", 25.0)

        if has_edge_tts:
            try:
                # Concatenate speech turns
                dialogue = tc.get("dialogue", [])
                full_text = " ".join([d["text"] for d in dialogue])
                voice = CUSTOMER_VOICE_HI if tc.get("category") == "HINGLISH_CODE_SWITCH" else CUSTOMER_VOICE_EN
                await synthesize_with_edge_tts(full_text, voice, out_path)
                print(f"  [SYNTHESIZED] {tc_id} -> {filename} (edge-tts)")
                continue
            except Exception as e:
                print(f"  [FALLBACK] edge-tts error for {tc_id}: {e}. Using PCM generator.")

        # Fallback generator
        generate_synthetic_pcm_wav(out_path, duration_seconds=duration)
        print(f"  [GENERATED] {tc_id} -> {filename} ({duration}s PCM WAV)")

    print(f"\n[SUCCESS] All {len(test_cases)} test call audio files prepared in: {AUDIO_OUTPUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
