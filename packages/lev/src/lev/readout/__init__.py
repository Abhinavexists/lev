"""Mode A reads label-token logits; Mode B learns scores over candidate text."""

from .mode_a import LabelTokenReadout
from .mode_b import CandidatePathReadout

__all__ = ["CandidatePathReadout", "LabelTokenReadout"]
