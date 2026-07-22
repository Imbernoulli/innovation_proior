# TIER: invalid
# References an unknown name in OUT (only S and v are legal there) -> the
# checker's strict DSL validator rejects it and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("KERNEL 1 / ( dist * dist )")
print("OUT v * wobble")
