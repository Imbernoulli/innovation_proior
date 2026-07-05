import sys

# Reservoir Dam Network -- geometric packing, format C (maximize sum of radii).
# Container: a rectangular valley floor [0,W] x [0,H].
# A protected wetland (circular obstacle) sits at (CX, CY) with radius Q; no
# reservoir may overlap it. Difficulty grows with the number of reservoirs N and
# with the wetland radius Q (both shrink the usable free area). Deterministic in
# testId only.
#
# Ladder invariant kept so the checker's row baseline stays feasible:
#   H/2 - Q >= W/N  (a single bottom row of N equal disks clears the wetland).
N_LADDER = [12, 18, 24, 30, 38, 46, 56, 66, 78, 90]
Q_LADDER = [0.15, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.35]

W = 2.0
H = 1.0
CX = 1.0
CY = 0.5

i = int(sys.argv[1])
idx = min(max(i, 1), len(N_LADDER)) - 1
N = N_LADDER[idx]
Q = Q_LADDER[idx]

print("%d %.6f %.6f %.6f %.6f %.6f" % (N, W, H, CX, CY, Q))
