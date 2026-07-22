# TIER: invalid
# Garbage: emits absurd, power-limit-violating and non-finite commitments.
import sys

data = sys.stdin.read().split("\n")
T = int(data[0].split()[0])
vals = []
for i in range(T):
    if i % 3 == 0:
        vals.append("1e30")
    elif i % 3 == 1:
        vals.append("nan")
    else:
        vals.append("-7.5e29")
sys.stdout.write(" ".join(vals) + "\n")
