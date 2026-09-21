"""Mode A: read the label-token logits at the answer boundary.

    ... Options:  A: refund   B: replace   C: info
    Answer:
            ^ the scored position -- we take the next-token logits here and keep
              only the ids for " A", " B", " C". Nothing is sampled or generated.

Zero added parameters, so this works on a stock checkpoint before any training.
That is why build step 1 in docs/ARCHITECTURE.md §5.9 needs no GPU time at all.

Verified against `Qwen/Qwen3.5-4B-Base`: the indexing below follows litjev's
`output.logits[i, len(suffix_ids[i]) - 1, candidate_ids[i]]` and produces sane
distributions on a real checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LabelTokenReadout:
    """Scores candidates by their single-token label id.

    `tokenizer` must be the one the router verified against -- using a different
    tokenizer silently changes which ids are read.
    """

    tokenizer: object
    prefix: str = " "

    def candidate_ids(self, codes: list[str]) -> list[int]:
        """Token id for each label code, as it appears after `Answer:`."""
        ids = []
        for code in codes:
            encoded = self.tokenizer.encode(self.prefix + code, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(
                    f"label {code!r} is {len(encoded)} tokens, not 1. "
                    "The router should have sent this question to Mode B."
                )
            ids.append(encoded[0])
        return ids

    def scores(
        self,
        logits: torch.Tensor,
        last_positions: torch.Tensor,
        candidate_ids: list[list[int]],
    ) -> list[torch.Tensor]:
        """Extract per-row candidate logits.

        Args:
            logits: `(batch, seq, vocab)` from one forward pass.
            last_positions: `(batch,)` index of each row's final input token --
                the position whose next-token distribution we read. Rows are
                right-padded, so this is *not* `seq - 1` for every row.
            candidate_ids: per-row token ids to keep.

        Returns:
            One 1-D tensor of raw logits per row. Raw, not softmaxed: temperature
            is applied later, per bucket (see `lev.calibrate`).
        """
        out = []
        for row, ids in enumerate(candidate_ids):
            position = int(last_positions[row])
            out.append(logits[row, position, ids])
        return out
