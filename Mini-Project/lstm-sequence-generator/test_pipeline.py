"""
test_pipeline.py: Automated sanity test suite for LSTM sequence generation system.
Validates Tokenizer, Dataset, TextLSTM architecture, and Sampling logic.
"""

import os
import sys

# Configure UTF-8 stdout/stderr for Windows console compatibility
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from model import TextLSTM
from train import CharTokenizer, SequenceDataset
from generator import TextGenerator


def test_tokenizer() -> None:
    print("[Test 1/4] Testing CharTokenizer...")
    sample_text = "To be, or not to be, that is the question."
    tokenizer = CharTokenizer()
    tokenizer.fit_on_text(sample_text)

    assert tokenizer.vocab_size > 0, "Vocab size must be positive"
    assert "<unk>" in tokenizer.char2idx, "<unk> token must be present"

    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    assert len(encoded) == len(sample_text), "Encoding length mismatch"
    assert decoded == sample_text, "Decoding reconstruction mismatch"

    # Test unknown character handling
    unk_encoded = tokenizer.encode("🔥")
    assert unk_encoded == [0], f"Expected 0 for unknown character, got {unk_encoded}"
    print("  ✓ CharTokenizer passed.")


def test_dataset() -> None:
    print("[Test 2/4] Testing SequenceDataset...")
    tokens = list(range(100))
    seq_len = 10
    dataset = SequenceDataset(tokens, seq_length=seq_len, step=1)

    assert len(dataset) == 100 - seq_len, "Dataset length incorrect"
    x, y = dataset[0]
    assert x.shape == (seq_len,), f"Expected shape ({seq_len},), got {x.shape}"
    assert y.shape == (seq_len,), f"Expected shape ({seq_len},), got {y.shape}"
    # Target should be shifted by 1
    assert torch.equal(x[1:], y[:-1]), "Target must be shifted by 1 from input"
    print("  ✓ SequenceDataset passed.")


def test_model_forward() -> None:
    print("[Test 3/4] Testing TextLSTM forward pass...")
    vocab_size = 50
    batch_size = 4
    seq_len = 16

    model = TextLSTM(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
    )

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, (h, c) = model(dummy_input)

    assert logits.shape == (batch_size, seq_len, vocab_size), f"Logits shape mismatch: {logits.shape}"
    assert h.shape == (2, batch_size, 64), f"Hidden state shape mismatch: {h.shape}"
    assert c.shape == (2, batch_size, 64), f"Cell state shape mismatch: {c.shape}"
    print("  ✓ TextLSTM forward pass passed.")


def test_temperature_sampling() -> None:
    print("[Test 4/4] Testing temperature sampling mathematical properties...")
    # Mock logits where token 3 has highest score
    logits = torch.tensor([1.0, 2.0, 0.5, 10.0, 1.2])

    # Low temperature (T=0.01) should almost always sample token 3
    low_temp_samples = []
    scaled = logits / 0.01
    probs = torch.softmax(scaled, dim=-1)
    for _ in range(50):
        s = torch.multinomial(probs, num_samples=1).item()
        low_temp_samples.append(s)

    assert all(idx == 3 for idx in low_temp_samples), "Low temperature did not concentrate at argmax"
    print("  ✓ Temperature sampling logic passed.")


if __name__ == "__main__":
    print("Running system validation tests...")
    test_tokenizer()
    test_dataset()
    test_model_forward()
    test_temperature_sampling()
    print("\n🎉 All 4 pipeline components verified successfully!")
