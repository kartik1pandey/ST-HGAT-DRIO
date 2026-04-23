"""Tests for GRUEncoder — unit tests and property-based tests.

Covers:
- Property 6: GRU sequence length invariance (Requirements 3.1, 3.3)
- Unit test: NaN rejection (Requirement 3.5)
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.model.gru_encoder import GRUEncoder


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 5
D_HIDDEN = 64


def make_encoder(input_dim: int = INPUT_DIM, d_hidden: int = D_HIDDEN) -> GRUEncoder:
    encoder = GRUEncoder(input_dim=input_dim, d_hidden=d_hidden)
    encoder.eval()
    return encoder


# ---------------------------------------------------------------------------
# Property 6: GRU Sequence Length Invariance
# Validates: Requirements 3.1, 3.3
# ---------------------------------------------------------------------------

# Feature: st-hgat-drio, Property 6: GRU sequence length invariance
@given(
    seq_len=st.integers(min_value=1, max_value=14),
    batch_size=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100)
def test_gru_sequence_length_invariance(seq_len: int, batch_size: int) -> None:
    """**Validates: Requirements 3.1, 3.3**

    For any sequence length in [1, 14], GRUEncoder must produce an output
    of shape [batch, d_hidden] regardless of the original sequence length.
    """
    encoder = make_encoder()
    x = torch.randn(batch_size, seq_len, INPUT_DIM)
    with torch.no_grad():
        out = encoder(x)
    assert out.shape == (batch_size, D_HIDDEN), (
        f"Expected output shape ({batch_size}, {D_HIDDEN}), got {out.shape} "
        f"for seq_len={seq_len}"
    )


# ---------------------------------------------------------------------------
# Unit test: NaN rejection (Requirement 3.5)
# ---------------------------------------------------------------------------

def test_nan_input_raises_value_error() -> None:
    """Passing a tensor with NaN values must raise ValueError."""
    encoder = make_encoder()
    x = torch.randn(4, 14, INPUT_DIM)
    x[0, 3, 2] = float("nan")  # inject a single NaN

    with pytest.raises(ValueError, match="NaN"):
        encoder(x)


def test_nan_all_raises_value_error() -> None:
    """A fully NaN tensor must also raise ValueError."""
    encoder = make_encoder()
    x = torch.full((2, 14, INPUT_DIM), float("nan"))

    with pytest.raises(ValueError):
        encoder(x)
