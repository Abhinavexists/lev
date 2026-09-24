"""Score candidate text with a shared attention head.

Question and candidate representations are projected to the same dimension.
Candidates attend to each other before scoring. Ordered targets also use
`ordinal_penalty` during training.
"""

from __future__ import annotations

import torch
from torch import nn


class CandidatePathReadout(nn.Module):
    """A candidate-set matching head shared by Choice, Score and Noul."""

    def __init__(self, hidden_size: int = 2560, proj_dim: int = 512, n_heads: int = 8):
        super().__init__()
        self.question_proj = nn.Linear(hidden_size, proj_dim, bias=False)
        self.candidate_proj = nn.Linear(hidden_size, proj_dim, bias=False)
        # Candidates attend to each other: "is this the best of these", not "is
        # this good in isolation".
        self.set_attention = nn.MultiheadAttention(proj_dim, n_heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(proj_dim)
        self.score = nn.Linear(proj_dim, 1, bias=False)

    def forward(
        self,
        question_repr: torch.Tensor,
        candidate_reprs: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            question_repr: `(batch, hidden)`.
            candidate_reprs: `(batch, n_candidates, hidden)`.
            candidate_mask: `(batch, n_candidates)`, True where padded.

        Returns:
            `(batch, n_candidates)` raw scores. Masked slots are -inf so they
            vanish under softmax regardless of the temperature applied later.
        """
        q = self.question_proj(question_repr).unsqueeze(1)
        c = self.candidate_proj(candidate_reprs)

        attended, _ = self.set_attention(c, c, c, key_padding_mask=candidate_mask)
        scores = self.score(self.norm(attended + q)).squeeze(-1)

        if candidate_mask is not None:
            scores = scores.masked_fill(candidate_mask, float("-inf"))
        return scores


def ordinal_penalty(scores: torch.Tensor, target_level: torch.Tensor) -> torch.Tensor:
    """Expected squared distance from the true level under the predicted distribution."""
    n_levels = scores.size(-1)
    levels = torch.arange(n_levels, device=scores.device, dtype=scores.dtype)
    distance = (levels.unsqueeze(0) - target_level.unsqueeze(1).to(scores.dtype)) ** 2
    return (scores.softmax(dim=-1) * distance).sum(dim=-1).mean()
