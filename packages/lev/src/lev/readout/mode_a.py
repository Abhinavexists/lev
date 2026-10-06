"""Map verified label codes to vocabulary ids at the answer boundary."""

from __future__ import annotations

from dataclasses import dataclass

from ..labels import TokenEncoder


@dataclass
class LabelTokenReadout:
    """Read label ids using the same tokenizer that verified the codes."""

    tokenizer: TokenEncoder
    prefix: str = " "

    def candidate_ids(self, codes: list[str]) -> list[int]:
        """Encode each label code at the configured answer boundary."""
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
