# TIER: invalid
"""Deliberately infeasible output: a malformed address token. Must score 0 regardless
of the input instance (checker's address regex rejects anything outside {'.', 0/1+})."""
import sys

sys.stdout.write("1\nBADADDR\n")
