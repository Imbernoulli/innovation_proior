# TIER: invalid
"""Ignores the table entirely and ships a plausible-looking but wrong one-line
guess (F(x,y) = x*y). Fails the exactness gate almost everywhere -> Ratio 0.0."""
import sys

sys.stdin.read()  # consume the instance, output is independent of it (garbage)

lines = [
    "1",
    "MUL x y",
    "OUT R1",
]
sys.stdout.write("\n".join(lines) + "\n")
