"""Mode A: read the label-token logits at the answer boundary.

    ... Options:  A: refund   B: replace   C: info
    Answer:
            ^ the scored position: its next-token logits, restricted to the ids
              for " A", " B", " C". Nothing is sampled or generated.

No added parameters, so it works on a stock checkpoint before any training.
Callers read `logits[row, last_position, candidate_ids]`, as litjev does.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LabelTokenReadout:
    """Maps label codes to the single token ids Mode A reads.

    `tokenizer` must be the one the router verified against; another tokenizer
    silently changes which ids are read.
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
