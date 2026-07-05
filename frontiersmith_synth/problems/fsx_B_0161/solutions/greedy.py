# TIER: greedy
# Grating superposition (phase-only).  To steer flow onto corridor (u,v) you add a
# linear phase ramp exp(2*pi*i*((u-c)/n * r + (v-c)/n * c)) across the grid; summing one
# ramp per corridor and keeping only the phase of the result aims flow at all corridors
# at once.  This clears the all-zero baseline handily (efficiency ~0.75-0.8), but the
# phase-only truncation throws away the amplitude the sum wanted, so the corridors come
# out very unequal (uniformity ~0.4-0.5).  It is a strong first guess with lots of
# headroom left for iterative refinement.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
n = inst["n"]
spots = inst["spots"]
c = n // 2

rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
field = np.zeros((n, n), dtype=complex)
for (u, v) in spots:
    ku = (u - c) / n
    kv = (v - c) / n
    field += np.exp(2j * np.pi * (ku * rr + kv * cc))

phases = np.angle(field)
print(json.dumps({"phases": phases.tolist()}))
