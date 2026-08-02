# TIER: invalid
# Emits an ABOVE line that references an unknown name and a THRESH line that
# (illegally) references Ta -- the strict AST whitelist in the checker
# rejects it and prints Ratio: 0.0.
import sys
sys.stdin.read()
print("THRESH Ta + 1.0")
print("BELOW banana * Ta")
print("ABOVE wobble(Ta)")
