# TIER: invalid
# Assigns the top tier M to every applicant, blowing the portfolio loss cap
# (and, on most instances, simply the M-token count vs N is fine but the
# aggregate loss check fails) -> Ratio: 0.0.
import sys

toks = sys.stdin.read().split()
N = int(toks[0]); M = int(toks[1])
print(" ".join(str(M) for _ in range(N)))
