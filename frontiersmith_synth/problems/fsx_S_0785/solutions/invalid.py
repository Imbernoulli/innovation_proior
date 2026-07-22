# TIER: invalid
# Emits a SLOT1 expression using true division "/", which is NOT in the
# allowed grammar (only // and % are permitted) -> the checker's strict AST
# validator rejects it and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("SLOT1 t / 2 + h1")
print("SLOT2 NONE")
