# TIER: trivial
"""Do nothing: spend none of the firebreak budget. Reproduces the evaluator's
own baseline() exactly, so this always scores ~0.1 per instance."""
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"cells": []}))
