"""LoRA bucket config for rank-based bucket scheduling.

Provides a dataclass that parses bucket boundary strings and maps LoRA ranks
to their ceiling bucket. This is pure CPU logic — no GPU dependencies.
"""

import sys
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LoRABucketConfig:
    """Immutable bucket configuration for LoRA rank-aware scheduling.

    Buckets define rank thresholds. Each incoming LoRA adapter whose rank
    exceeds a bucket boundary is assigned to the next larger bucket (the
    "ceiling"). Base model (rank 0) always returns 0.
    """

    buckets: Tuple[int, ...]

    @classmethod
    def from_string(cls, s: str) -> "LoRABucketConfig":
        """Parse a comma-separated bucket string.

        Example: "0,8,16,32,64" → (0, 8, 16, 32, 64)

        Each part is stripped of surrounding whitespace before parsing.
        The result is sorted and deduplicated.

        Raises:
            ValueError: If the string is empty, contains non-numeric parts,
                includes negative values, does not contain 0, or has a
                trailing comma.
        """
        if not s.strip():
            raise ValueError("Bucket string must not be empty")

        if s.strip().endswith(","):
            raise ValueError(
                f"Bucket string must not end with a trailing comma: {s!r}"
            )

        parts = [p.strip() for p in s.split(",")]
        parsed: list[int] = []
        for p in parts:
            if not p:
                raise ValueError(
                    f"Bucket string contains an empty part: {s!r}"
                )
            try:
                val = int(p)
            except ValueError:
                raise ValueError(
                    f"Bucket values must be integers, got {p!r} in {s!r}"
                ) from None
            if val < 0:
                raise ValueError(
                    f"Bucket values must be non-negative, got {val} in {s!r}"
                )
            parsed.append(val)

        unique = tuple(sorted(set(parsed)))

        if 0 not in unique:
            raise ValueError(
                f"Bucket config must contain 0, got {s!r}"
            )

        return cls(buckets=unique)

    def get_ceiling(self, rank: int) -> int:
        """Return the ceiling bucket for *rank*.

        - rank == 0 → 0 (base model bypass)
        - rank > 0 → the smallest bucket value strictly greater than rank
        - rank exceeds the largest bucket → sys.maxsize
        """
        if rank == 0:
            return 0
        for b in self.buckets:
            if b > rank:
                return b
        return sys.maxsize
