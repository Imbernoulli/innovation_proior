#!/usr/bin/env python3
# Trivial baseline: select ALL sites. Always feasible by construction.
import sys
with open(sys.argv[1]) as f:
    toks = f.read().split()
n = int(toks[0]); m = int(toks[1])
print(m, *range(m))
