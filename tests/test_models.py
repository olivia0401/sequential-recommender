"""Regression tests for the model code (CPU, tiny sizes, no dataset needed)."""
import torch

from src.data import PAD
from src.models import SASRec


def test_sasrec_left_padded_input_has_no_nan():
    # Regression for the masking bug: a left-padded history used to produce nan
    # at the final position, and nan scores ranked every such user as rank 1.
    torch.manual_seed(0)
    model = SASRec(n_items=20, max_len=10).eval()
    x = torch.tensor([
        [PAD] * 7 + [3, 4, 5],   # typical left-padded user
        [PAD] * 9 + [7],         # single-item history
        [PAD] * 10,              # empty history
        list(range(1, 11)),      # no padding
    ])
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()


def test_sasrec_batch_matches_single_sample():
    # The per-sample attention mask is expanded per head and flattened; if that
    # ordering were wrong, one user's padding would mask another user's keys.
    torch.manual_seed(0)
    model = SASRec(n_items=20, max_len=10, heads=2).eval()
    rows = [
        [PAD] * 7 + [3, 4, 5],
        list(range(1, 11)),
        [PAD] * 9 + [7],
    ]
    with torch.no_grad():
        batched = model(torch.tensor(rows))[:, -1]
        single = torch.stack([model(torch.tensor([r]))[0, -1] for r in rows])
    assert torch.allclose(batched, single, atol=1e-5)
