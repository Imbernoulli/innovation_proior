# TIER: invalid
# Emits a program that violates the grammar in two ways at once: it calls an
# unknown function and references an unknown name -> the checker's strict
# DSL validator rejects it and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("OUT wobble ( x ) + banana")
