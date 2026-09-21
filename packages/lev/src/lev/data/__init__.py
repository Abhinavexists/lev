"""Training data assembly, and the contamination guard that gates it."""

from .contamination import (
    BLOCKED_SUBSETS,
    ContaminationError,
    assert_clean,
    check_mixture,
)

__all__ = ["BLOCKED_SUBSETS", "ContaminationError", "assert_clean", "check_mixture"]
