# TIER: invalid
"""Structurally infeasible: predicts a strictly negative wait time for every
possible non-negative load, which the checker must reject (wait times cannot
be negative)."""
print("0 - L - 1")
