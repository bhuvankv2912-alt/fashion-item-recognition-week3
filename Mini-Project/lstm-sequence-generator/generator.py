"""
generator.py: Sequence Generation & Temperature-Based Sampling Engine.

Features:
- Encapsulates TextLSTM model and CharTokenizer for clean, decoupled inference.
- Temperature-scaled softmax sampling: logits / temperature.
- Optional Top-K nucleus filtering to prevent low-probability gibberish.
- Seed conditioning and recurrent hidden state management.
- Diagnostic distribution inspection for visualization in UI (top token probabilities).
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from model import TextLSTM
from train import CharTokenizer


class TextGenerator:
    """
    Inference and sampling engine for trained recurrent LSTM sequence models.
    """

    def __init__(
        self,
        model_path: str = os.path.join("models", "model.pt"),
        vocab_path: str = os.path.join("models", "vocab.json"),
        device: Optional[str] = None,
    ) -> None:
        """
        Load trained model weights and tokenizer mapping from disk.

        Args:
            model_path (str): Path to saved PyTorch checkpoint (.pt).
            vocab_path (str): Path to saved vocabulary mapping (.json).
            device (Optional[str]): Execution device ('cpu' or 'cuda').
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary mapping file not found at: {vocab_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint file not found at: {model_path}")

        # 1. Load Tokenizer
        self.tokenizer = CharTokenizer.load(vocab_path)

        # 2. Load Model Checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get("config", {})

        self.embedding_dim = config.get("embedding_dim", 128)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.2)
        self.seq_length = config.get("seq_length", 64)
        self.history = checkpoint.get("history", [])

        self.model = TextLSTM(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(
            f"[Generator] Loaded checkpoint from {model_path} "
            f"(Epoch: {checkpoint.get('epoch', 'N/A')}, Val Loss: {checkpoint.get('val_loss', 0.0):.4f})"
        )

    def sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> int:
        """
        Apply temperature scaling and optional top-k filtering, then sample from
        the categorical multinomial distribution.

        Args:
            logits (torch.Tensor): Unnormalized scores for next token of shape (vocab_size,).
            temperature (float): Softmax temperature (> 0).
                - Low temp (< 0.5): High confidence, peaky, deterministic.
                - Normal temp (0.7 - 0.9): Balanced, creative yet coherent.
                - High temp (> 1.2): High entropy, random, diverse.
            top_k (int): If > 0, retain only the top k tokens with highest logits.

        Returns:
            int: Sampled token ID.
        """
        temp = max(float(temperature), 1e-4)
        scaled_logits = logits / temp

        # Optional Top-K Filtering
        if top_k > 0 and top_k < scaled_logits.size(-1):
            top_values, _ = torch.topk(scaled_logits, top_k)
            min_top_value = top_values[-1]
            scaled_logits = torch.where(
                scaled_logits < min_top_value,
                torch.tensor(float("-inf"), device=scaled_logits.device),
                scaled_logits,
            )

        probabilities = F.softmax(scaled_logits, dim=-1)

        # Numerical safety check
        if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
            return int(torch.argmax(logits).item())

        sampled_idx = torch.multinomial(probabilities, num_samples=1).item()
        return sampled_idx

    def generate(
        self,
        seed_prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_k: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate continuous text given a seed prompt.

        Args:
            seed_prompt (str): Initial context text to prime the LSTM.
            max_length (int): Number of new tokens (characters) to generate.
            temperature (float): Sampling randomness scaling factor.
            top_k (int): Top-K candidate restriction (0 = disabled).

        Returns:
            Dict[str, Any]: {
                "generated_text": Full string (seed + new content),
                "new_tokens": Just the newly generated content,
                "token_count": Count of generated tokens,
                "temperature": Temperature used,
                "top_k": Top-k used,
                "final_distribution": Top 10 tokens and probabilities at the final step.
            }
        """
        if not seed_prompt:
            seed_prompt = " "

        encoded_seed = self.tokenizer.encode(seed_prompt)
        input_tensor = torch.tensor(encoded_seed, dtype=torch.long, device=self.device).unsqueeze(0)

        generated_chars: List[str] = []
        hidden = None

        with torch.no_grad():
            # 1. Warm-up phase: Feed seed prompt through model to establish recurrent state
            logits, hidden = self.model(input_tensor, hidden)
            last_logits = logits[0, -1, :]

            # Sample the first character immediately following seed
            next_token_id = self.sample_next_token(
                last_logits, temperature=temperature, top_k=top_k
            )
            next_char = self.tokenizer.idx2char.get(next_token_id, "")
            generated_chars.append(next_char)

            curr_input = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)

            # 2. Autoregressive Rollout Loop
            for _ in range(max(1, max_length - 1)):
                logits, hidden = self.model(curr_input, hidden)
                step_logits = logits[0, -1, :]

                next_token_id = self.sample_next_token(
                    step_logits, temperature=temperature, top_k=top_k
                )
                next_char = self.tokenizer.idx2char.get(next_token_id, "")
                generated_chars.append(next_char)

                curr_input = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)

            # Compute distribution of candidate tokens for final token step for visualization
            scaled_final = step_logits / max(temperature, 1e-4)
            final_probs = F.softmax(scaled_final, dim=-1)
            top_probs, top_indices = torch.topk(final_probs, k=min(10, len(final_probs)))

            top_candidates = []
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                char_repr = self.tokenizer.idx2char.get(idx, "")
                # Format whitespace and control characters cleanly
                display_char = (
                    "\\n (newline)" if char_repr == "\n"
                    else "␣ (space)" if char_repr == " "
                    else char_repr
                )
                top_candidates.append({
                    "token": display_char,
                    "raw_token": char_repr,
                    "probability": round(prob * 100, 2),
                })

        new_text = "".join(generated_chars)
        full_text = seed_prompt + new_text

        return {
            "seed_prompt": seed_prompt,
            "generated_text": full_text,
            "new_text": new_text,
            "token_count": len(generated_chars),
            "temperature": temperature,
            "top_k": top_k,
            "top_candidates": top_candidates,
        }

    def inspect_distribution(
        self,
        prompt: str,
        temperature: float = 1.0,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Inspect next-token probability distribution given a prompt and temperature.
        Useful for interactive temperature sensitivity comparison in the UI.
        """
        encoded = self.tokenizer.encode(prompt if prompt else " ")
        input_tensor = torch.tensor(encoded, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(input_tensor)
            last_logits = logits[0, -1, :]
            scaled = last_logits / max(float(temperature), 1e-4)
            probs = F.softmax(scaled, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=min(top_n, len(probs)))

            results = []
            for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
                char = self.tokenizer.idx2char.get(idx, "")
                display_char = (
                    "\\n (newline)" if char == "\n"
                    else "␣ (space)" if char == " "
                    else char
                )
                results.append({
                    "Token": display_char,
                    "Probability (%)": round(p * 100, 2),
                })
            return results
