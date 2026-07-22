# TIER: invalid
# Emits the same (short) codeword "0" for every symbol -- badly violates
# prefix-freeness (every codeword is a duplicate/prefix of every other) so the
# checker must reject it with Ratio: 0.0.
import sys

toks = sys.stdin.read().split()
n = int(toks[0])
print(" ".join(["0"] * n))
