# TIER: invalid
"""Emits garbage: negative payload and a truck index out of range."""
import sys

_ = sys.stdin.read()
sys.stdout.write("0\n1\n1 999 0 -5.0\n0\n")
