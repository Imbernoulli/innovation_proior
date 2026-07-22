# TIER: invalid
"""Deliberately invalid: massively over budget AND full of duplicate cells.
Must score 0.0 on every instance."""
import sys, json

inst = json.load(sys.stdin)
budget = inst.get("budget", 10)
cells = [[0, 0]] * (budget * 5 + 50)
print(json.dumps({"cells": cells}))
