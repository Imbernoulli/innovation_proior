# TIER: invalid
# Places a well far outside the field (containment violation) -> scores 0.
import sys

t = sys.stdin.read().split()
N = int(t[0])
print(N)
out = []
for k in range(N):
    # every well parked at (5, 5), well outside [0,1]^2
    out.append("5.0 5.0")
sys.stdout.write("\n".join(out) + "\n")
