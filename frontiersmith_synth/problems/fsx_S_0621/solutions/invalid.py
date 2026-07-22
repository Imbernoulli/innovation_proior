# TIER: invalid
# Halt immediately: performs no multiplies, so it fails to compute the required
# nonzero blocks of every pattern -> the checker must score 0.
import sys
sys.stdin.read()
sys.stdout.write("1\nH\n")
