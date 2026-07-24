# TIER: invalid
# Emits a program referencing an unknown function/name -> the checker's strict
# DSL validator rejects it and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("V1 wobble ( v1 ) + banana")
print("V2 v2")
