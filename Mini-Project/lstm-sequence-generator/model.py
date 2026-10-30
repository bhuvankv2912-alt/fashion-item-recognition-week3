"""
model.py: Deep Learning LSTM Architecture for Sequence Generation.
Provides the PyTorch LSTM neural network module with embedding, stacked LSTM cells,
dropout regularization, and vocabulary projection.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any


class TextLSTM(nn.Module):
    """
    Multi-layer Long Short-Term Memory (LSTM) recurrent neural network
    for character/token-level sequence modeling and generation.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """
        Initialize the LSTM Sequence Model.

        Args:
            vocab_size (int): Total number of unique tokens in vocabulary.
            embedding_dim (int): Dimensionality of token embedding space.
            hidden_dim (int): Number of hidden features per LSTM cell.
            num_layers (int): Number of recurrent LSTM layers stacked.
            dropout (float): Dropout probability between LSTM layers (regularization).
        """
        super(TextLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # 1. Embedding Layer: Projects discrete token indices into continuous dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. Recurrent LSTM Stack: Captures long-range temporal sequence dependencies
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 3. Dropout regularizer before dense layer
        self.dropout = nn.Dropout(p=dropout)

        # 4. Dense projection layer: Maps hidden state to vocabulary logits
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward computation pass through embedding -> LSTM -> Linear projection.

        Args:
            x (torch.Tensor): Input token indices of shape (batch_size, seq_len).
            hidden (Optional[Tuple[torch.Tensor, torch.Tensor]]): Initial (h_0, c_0) states.

        Returns:
            logits (torch.Tensor): Unnormalized log-probabilities of shape (batch_size, seq_len, vocab_size).
            hidden (Tuple[torch.Tensor, torch.Tensor]): Updated (h_t, c_t) recurrent states.
        """
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # Pass through stacked LSTM
        out, hidden = self.lstm(embedded, hidden)

        # Apply dropout to LSTM outputs
        out = self.dropout(out)

        # Project to vocabulary size: (batch_size, seq_len, vocab_size)
        logits = self.fc(out)

        return logits, hidden

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate zero-initialized hidden and cell states for batch processing.

        Args:
            batch_size (int): Current batch size.
            device (torch.device): Target device (CPU or CUDA).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (h_0, c_0) zero tensors.
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0)

    def get_summary(self) -> Dict[str, Any]:
        """
        Return architectural parameter details and counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
