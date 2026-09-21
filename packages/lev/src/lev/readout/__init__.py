"""The two readout mechanisms. See docs/ARCHITECTURE.md §5.3.

Both turn one forward pass into a probability distribution over candidates, and
neither generates a token. They differ in *what* gets scored:

  mode_a  the logit of a single label token (`A`, `B`, ...) at the answer boundary
  mode_b  a learned match between the question and each candidate's own text

Mode A needs no parameters and no training. Mode B needs a trained head but has no
option ceiling. The router picks per question; a request can use both at once.
"""

from .mode_a import LabelTokenReadout
from .mode_b import CandidatePathReadout

__all__ = ["CandidatePathReadout", "LabelTokenReadout"]
