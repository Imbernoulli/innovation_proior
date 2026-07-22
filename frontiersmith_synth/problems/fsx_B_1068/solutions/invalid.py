# TIER: invalid
"""Emits an out-of-range word-index poem -- must be rejected by the checker (Ratio: 0.0)."""
N_LINES = 14
N_SLOTS = 5

for _ in range(N_LINES):
    print(" ".join(["999999"] * N_SLOTS))
