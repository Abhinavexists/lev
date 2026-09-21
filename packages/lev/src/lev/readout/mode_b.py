"""Mode B: score each candidate's own text against the question.

This is what removes the option ceiling. Instead of projecting onto one label token,
we embed each candidate's text and learn a match against the question's
representation. Candidates are scored as a *set*, in one backbone forward, following
NanoJev ("6 states, 18 questions, 44 candidate paths, 1 backbone forward").

    question repr ─┐
                   ├─> set attention over candidates ─> shared scalar head ─> score
    candidate reprs ┘

Two things this buys over Mode A: no single-token constraint, and the candidate's
semantics reach the scorer directly rather than only via the prompt.

One thing it costs: ordinality is no longer free. Under Mode A a Score's levels are
unordered symbols and `sum(i * p_i)` works because ordering lives in the prompt text.
Here nothing forces level i+1 to score above level i, so `ordinal_penalty` adds an
explicit monotonicity term and `lev.metrics` tracks `ordinal_mae` separately --
accuracy hides this failure mode entirely. decider reports the same metric for the
same reason.

NOT YET RUN ON HARDWARE. Parameter counts and shapes are designed, not measured.
"""

from __future__ import annotations

import torch
from torch import nn


class CandidatePathReadout(nn.Module):
    """Shared matching head over a candidate set. ~30M params at hidden=2560.

    One head serves Choice, Score and Noul: the question type changes how the
    resulting distribution is *interpreted*, not how it is produced.
    """

    def __init__(self, hidden_size: int = 2560, proj_dim: int = 512, n_heads: int = 8):
        super().__init__()
        self.question_proj = nn.Linear(hidden_size, proj_dim, bias=False)
        self.candidate_proj = nn.Linear(hidden_size, proj_dim, bias=False)
        # Candidates attend to each other, so a score can depend on the alternatives
        # -- "is this the best of these" rather than "is this good in isolation".
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
        # Condition every candidate on the question before scoring.
        scores = self.score(self.norm(attended + q)).squeeze(-1)

        if candidate_mask is not None:
            scores = scores.masked_fill(candidate_mask, float("-inf"))
        return scores


def ordinal_penalty(scores: torch.Tensor, target_level: torch.Tensor) -> torch.Tensor:
    """Penalise probability mass by its distance from the true level.

    Cross-entropy treats "predicted level 0 when the answer was 4" exactly as badly
    as "predicted level 3" -- which is wrong for an ordered scale, and is the
    failure Mode A avoids structurally. Weighting by squared level distance restores
    the ordering the prompt used to supply.
    """
    n_levels = scores.size(-1)
    levels = torch.arange(n_levels, device=scores.device, dtype=scores.dtype)
    distance = (levels.unsqueeze(0) - target_level.unsqueeze(1).to(scores.dtype)) ** 2
    return (scores.softmax(dim=-1) * distance).sum(dim=-1).mean()
