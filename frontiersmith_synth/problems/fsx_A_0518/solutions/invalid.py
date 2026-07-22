# TIER: invalid
# Emits a program referencing an unknown function/name -> the checker's strict
# DSL validator rejects it and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("OUT wobble ( d ) + banana")
