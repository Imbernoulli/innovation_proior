# TIER: invalid
"""Emits a syntactically out-of-grammar artifact (uses '**', which the
checker's grammar does not allow) -- must score 0 on every case."""
import sys

sys.stdin.read()
print("x ** 2")
