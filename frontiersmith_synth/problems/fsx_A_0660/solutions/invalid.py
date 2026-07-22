# TIER: invalid
"""Deliberately broken: crashes before producing any answer."""
import sys, json
inst = json.load(sys.stdin)
raise RuntimeError("this policy forgot how dams work")
