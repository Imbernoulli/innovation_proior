# TIER: invalid
"""Deliberately malformed program: references a register outside NVEC's
declared range, so the checker must reject it before any scoring."""
import sys

sys.stdin.read()  # consume input (ignored)
print("NVEC 3")
print("MATVEC 1 0")
print("MATVEC 7 1")   # register 7 is out of range for NVEC 3
print("OUTPUT 7")
