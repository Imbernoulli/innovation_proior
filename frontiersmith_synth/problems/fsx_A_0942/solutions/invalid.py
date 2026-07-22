# TIER: invalid
# References an undefined name -> the checker's strict expression validator
# rejects it (only the variable L is allowed) and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("phantom_charge*L")
