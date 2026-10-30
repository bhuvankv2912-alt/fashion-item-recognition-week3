"""
train.py: Data Preprocessing, Tokenization, and LSTM Model Training Pipeline.

Features:
- Character-level tokenization with bidirectional mapping (char2idx, idx2char).
- Out-of-vocabulary handling with reserved <unk> token.
- Sliding window PyTorch Dataset with autoregressive sequence targets.
- Multi-layer LSTM model training with gradient clipping and Perplexity tracking.
- Interactive live generation sample after each epoch.
- Serializes trained model checkpoint (model.pt) and vocabulary mapping (vocab.json).
"""

import os
import sys
import json
import math
import time
import argparse
from typing import Dict, List, Tuple

# Configure UTF-8 stdout/stderr for Windows console compatibility
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import TextLSTM


# ============================================================================
# 1. Dataset & Tokenizer Definition
# ============================================================================

class CharTokenizer:
    """
    Bidirectional character-level tokenizer with out-of-vocabulary support.
    """

    UNK_TOKEN = "<unk>"

    def __init__(self) -> None:
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        self.vocab_size: int = 0

    def fit_on_text(self, text: str) -> None:
        """
        Build vocabulary from unique characters in raw text corpus.
        """
        unique_chars = sorted(list(set(text)))
        # Reserve index 0 for unknown character token
        self.char2idx = {self.UNK_TOKEN: 0}
        self.idx2char = {0: self.UNK_TOKEN}

        for idx, char in enumerate(unique_chars, start=1):
            self.char2idx[char] = idx
            self.idx2char[idx] = char

        self.vocab_size = len(self.char2idx)
        print(f"[Tokenizer] Vocabulary built. Total unique tokens: {self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        """Convert a string of text into a list of integer token IDs."""
        unk_idx = self.char2idx.get(self.UNK_TOKEN, 0)
        return [self.char2idx.get(char, unk_idx) for char in text]

    def decode(self, indices: List[int]) -> str:
        """Convert a list of integer token IDs back into string characters."""
        return "".join([self.idx2char.get(idx, "") for idx in indices])

    def save(self, filepath: str) -> None:
        """Serialize tokenizer mappings to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "char2idx": self.char2idx,
            "idx2char": {str(k): v for k, v in self.idx2char.items()},
            "vocab_size": self.vocab_size,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Tokenizer] Saved vocabulary mapping to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        """Load tokenizer mappings from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.char2idx = data["char2idx"]
        tokenizer.idx2char = {int(k): v for k, v in data["idx2char"].items()}
        tokenizer.vocab_size = data["vocab_size"]
        return tokenizer


class SequenceDataset(Dataset):
    """
    PyTorch Dataset implementing sliding-window sequence slicing.
    Each input sequence X of length seq_length produces target sequence Y
    shifted right by one token: X = text[i : i + seq_len], Y = text[i+1 : i + seq_len + 1].
    """

    def __init__(self, encoded_tokens: List[int], seq_length: int, step: int = 1) -> None:
        self.seq_length = seq_length
        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []

        total_length = len(encoded_tokens)
        for i in range(0, total_length - seq_length, step):
            seq_x = encoded_tokens[i : i + seq_length]
            seq_y = encoded_tokens[i + 1 : i + seq_length + 1]
            self.inputs.append(torch.tensor(seq_x, dtype=torch.long))
            self.targets.append(torch.tensor(seq_y, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


# ============================================================================
# 2. Live Sampling Utility for Epoch Inspection
# ============================================================================

def sample_generation(
    model: TextLSTM,
    tokenizer: CharTokenizer,
    seed_prompt: str,
    gen_length: int = 80,
    temperature: float = 0.7,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Quick generation helper to inspect training progress after each epoch."""
    model.eval()
    with torch.no_grad():
        encoded = tokenizer.encode(seed_prompt)
        if not encoded:
            encoded = [1]
        input_seq = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
        generated = list(seed_prompt)

        hidden = None
        # Prime the LSTM hidden state on the prompt
        logits, hidden = model(input_seq, hidden)
        last_logit = logits[:, -1, :] / max(temperature, 1e-4)
        probs = torch.softmax(last_logit, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(tokenizer.idx2char.get(next_token, ""))

        # Iteratively generate characters
        curr_token = torch.tensor([[next_token]], dtype=torch.long, device=device)
        for _ in range(gen_length):
            logits, hidden = model(curr_token, hidden)
            scaled_logits = logits[:, -1, :] / max(temperature, 1e-4)
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(tokenizer.idx2char.get(next_token, ""))
            curr_token = torch.tensor([[next_token]], dtype=torch.long, device=device)

    model.train()
    return "".join(generated)


# ============================================================================
# 3. Training Loop Engine
# ============================================================================

def train_model(args: argparse.Namespace) -> None:
    """
    Main execution pipeline for loading data, tokenizing, building LSTM,
    executing training epochs, and checkpointing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print("=" * 70)
    print("  🚀 LSTM SEQUENCE GENERATION TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dataset path: {args.data}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | Seq len: {args.seq_length}")
    print(f"Hidden dim: {args.hidden_dim} | Embedding dim: {args.embedding_dim} | LR: {args.lr}")

    # Step 1: Read raw corpus
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Corpus file not found at: {args.data}")

    with open(args.data, "r", encoding="utf-8") as f:
        corpus = f.read()

    print(f"[Dataset] Total raw characters in corpus: {len(corpus):,}")

    # Step 2: Fit Tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit_on_text(corpus)
    encoded_corpus = tokenizer.encode(corpus)

    # Save tokenizer vocabulary mapping
    vocab_save_path = os.path.join(args.model_dir, "vocab.json")
    tokenizer.save(vocab_save_path)

    # Step 3: Create Sliding Window Dataset & DataLoader
    # Train/Validation split (90% / 10%)
    split_idx = int(0.9 * len(encoded_corpus))
    train_tokens = encoded_corpus[:split_idx]
    val_tokens = encoded_corpus[split_idx:]

    train_dataset = SequenceDataset(train_tokens, seq_length=args.seq_length, step=args.step)
    val_dataset = SequenceDataset(val_tokens, seq_length=args.seq_length, step=args.step)

    print(f"[DataLoader] Total training sequences: {len(train_dataset):,}")
    print(f"[DataLoader] Total validation sequences: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Step 4: Initialize LSTM Model
    model = TextLSTM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    summary = model.get_summary()
    print(f"[Model] Initialized TextLSTM: {summary['total_parameters']:,} total parameters.")

    # Step 5: Optimizer & Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    model_save_path = os.path.join(args.model_dir, "model.pt")
    history: List[Dict[str, float]] = []

    # Step 6: Epoch Training Loop
    start_total_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits, _ = model(x_batch)

            # Flatten logits and targets for cross-entropy
            # logits: (batch_size, seq_length, vocab_size) -> (batch_size * seq_length, vocab_size)
            # targets: (batch_size, seq_length) -> (batch_size * seq_length)
            loss = criterion(
                logits.view(-1, tokenizer.vocab_size),
                y_batch.view(-1),
            )

            loss.backward()

            # Gradient clipping to stabilize recurrent propagation
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        train_ppl = math.exp(min(avg_train_loss, 20.0))

        # Validation Pass
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                val_logits, _ = model(x_val)
                val_loss = criterion(
                    val_logits.view(-1, tokenizer.vocab_size),
                    y_val.view(-1),
                )
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        val_ppl = math.exp(min(avg_val_loss, 20.0))

        scheduler.step(avg_train_loss)
        epoch_time = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_ppl": train_ppl,
            "val_loss": avg_val_loss,
            "val_ppl": val_ppl,
        })

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.2f}) | "
            f"Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.2f}) | "
            f"Time: {epoch_time:.2f}s",
            flush=True,
        )

        # Inspect live generated snippet every 3 epochs or on final epoch
        if epoch % 3 == 0 or epoch == args.epochs:
            seed_sample = "Shall I compare"
            sample_text = sample_generation(
                model=model,
                tokenizer=tokenizer,
                seed_prompt=seed_sample,
                gen_length=90,
                temperature=0.7,
                device=device,
            )
            print(f"  --> Sample Preview: {repr(sample_text)}", flush=True)

        # Save Checkpoint whenever training loss improves or on final epoch
        if avg_train_loss < best_train_loss or epoch == args.epochs:
            best_train_loss = avg_train_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "config": {
                    "vocab_size": tokenizer.vocab_size,
                    "embedding_dim": args.embedding_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "seq_length": args.seq_length,
                },
                "history": history,
            }
            torch.save(checkpoint, model_save_path)

    total_duration = time.time() - start_total_time
    print("=" * 70)
    print(f"✅ Training completed in {total_duration:.1f}s.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model checkpoint saved to: {model_save_path}")
    print(f"Vocabulary mapping saved to: {vocab_save_path}")
    print("=" * 70)


# ============================================================================
# 4. CLI Argument Parser
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM Sequence Generator")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("data", "poetry.txt"),
        help="Path to raw text corpus file",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save model checkpoint and vocabulary file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for DataLoader",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=64,
        help="Length of sliding sequence window",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=2,
        help="Stride step size for sliding window sequence extraction",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Dimension of token embeddings",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden state dimension of LSTM cells",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of recurrent LSTM layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout regularization rate",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.003,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=5.0,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Hardware execution device",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    train_model(cli_args)
