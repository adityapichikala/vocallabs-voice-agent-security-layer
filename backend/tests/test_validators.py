"""Unit tests for centralized validators."""
import unittest
import struct
from backend.app.validators import (
    AudioValidator, KnowledgeBaseValidator, LLMOutputValidator,
    PromiseStateValidator, PipelineStateValidator, ValidationError
)
from backend.app.models import PromiseState, CallState

class TestValidators(unittest.TestCase):
    def test_audio_validator_rejects_empty_bytes(self):
        with self.assertRaises(ValidationError):
            AudioValidator.validate_file("empty.wav", b"")

    def test_audio_validator_rejects_invalid_extension(self):
        with self.assertRaises(ValidationError):
            AudioValidator.validate_file("malicious.exe", b"RIFF....WAVE....")

    def test_audio_validator_validates_pcm_wav(self):
        # Construct minimal 44-byte WAV header
        header = bytearray(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
        # Add 100 samples of non-zero audio
        for i in range(100):
            header.extend(struct.pack("<h", 1500))
        
        res = AudioValidator.validate_file("sample.wav", bytes(header))
        self.assertTrue(res["valid"])
        self.assertEqual(res["sample_rate"], 16000)

    def test_audio_validator_handles_truncated_chunk_defensively(self):
        # Header claiming data chunk of 10000 bytes but only 50 bytes provided
        header = bytearray(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x10\x27\x00\x00")
        for i in range(25):
            header.extend(struct.pack("<h", 1200))
        
        res = AudioValidator.validate_file("truncated.wav", bytes(header))
        self.assertTrue(res["valid"])

    def test_llm_output_validator_sanitizes_markdown_json(self):
        raw = """```json
        {
            "turn_id": "turn_123",
            "flags": [{"type": "HALLUCINATION", "severity": "CRITICAL", "detail": "Fake 50% discount"}],
            "promises": [{"who": "agent", "action": "Callback", "target_entity": "CALLBACK", "deadline_raw": "5 PM", "is_authorized": true}],
            "confidence": 0.95,
            "reasoning": "Agent hallucinated discount."
        }
        ```"""
        parsed = LLMOutputValidator.parse_and_validate(raw, "turn_123")
        self.assertEqual(parsed["turn_id"], "turn_123")
        self.assertEqual(len(parsed["flags"]), 1)
        self.assertEqual(parsed["flags"][0]["type"], "HALLUCINATION")
        self.assertEqual(parsed["flags"][0]["severity"], "CRITICAL")
        self.assertEqual(len(parsed["promises"]), 1)
        self.assertEqual(parsed["confidence"], 0.95)

    def test_promise_state_validator_transitions(self):
        # Valid: PENDING -> FULFILLED
        self.assertTrue(PromiseStateValidator.validate_transition(PromiseState.PENDING.value, PromiseState.FULFILLED.value))
        
        # Invalid: FULFILLED -> PENDING
        with self.assertRaises(ValidationError):
            PromiseStateValidator.validate_transition(PromiseState.FULFILLED.value, PromiseState.PENDING.value)

    def test_pipeline_state_validator_transitions(self):
        # Valid sequence: UPLOADED -> TRANSCRIBING -> SCORING -> COMPLETE
        self.assertTrue(PipelineStateValidator.validate_transition(CallState.UPLOADED.value, CallState.TRANSCRIBING.value))
        self.assertTrue(PipelineStateValidator.validate_transition(CallState.TRANSCRIBING.value, CallState.SCORING.value))
        self.assertTrue(PipelineStateValidator.validate_transition(CallState.SCORING.value, CallState.COMPLETE.value))
        
        # Invalid: COMPLETE -> TRANSCRIBING
        with self.assertRaises(ValidationError):
            PipelineStateValidator.validate_transition(CallState.COMPLETE.value, CallState.TRANSCRIBING.value)

if __name__ == "__main__":
    unittest.main()
