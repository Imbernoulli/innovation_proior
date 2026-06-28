#!/usr/bin/env python3
"""Verbose feasibility checker: replays a solution and reports the FIRST violation."""
import sys
from collections import deque
DIRS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

def read_instance(path):
    with open(path) as f:
        toks = f.read().split('\n')
    idx = 0
    while toks[idx].strip() == '':
        idx += 1
    H, W, k = map(int, toks[idx].split()); idx += 1
    grid = []
    for _ in range(H):
        row = toks[idx]; idx += 1
        if len(row) < W: row = row + '.' * (W - len(row))
        grid.append(list(row[:W]))
    starts = []; targets = []
    for _ in range(k):
        while toks[idx].strip() == '':
            idx += 1
        sr, sc, tr, tc = map(int, toks[idx].split()); idx += 1
        starts.append((sr, sc)); targets.append((tr, tc))
    return H, W, k, grid, starts, targets

inst, sol = sys.argv[1], sys.argv[2]
H, W, k, grid, starts, targets = read_instance(inst)
pos = [list(p) for p in starts]
occ = {}
for i, (r, c) in enumerate(pos):
    occ[(r, c)] = i
with open(sol) as f:
    toks = f.read().split()
L = int(toks[0])
p = 1
for step in range(L):
    i = int(toks[p]); d = toks[p+1]; p += 2
    dr, dc = DIRS[d]
    r, c = pos[i]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < H and 0 <= nc < W):
        print(f"VIOLATION step {step}: token {i} moved off grid to ({nr},{nc})"); sys.exit()
    if grid[nr][nc] == '#':
        print(f"VIOLATION step {step}: token {i} moved into wall ({nr},{nc})"); sys.exit()
    if (nr, nc) in occ:
        print(f"VIOLATION step {step}: token {i} -> ({nr},{nc}) occupied by token {occ[(nr,nc)]}"); sys.exit()
    del occ[(r, c)]
    occ[(nr, nc)] = i
    pos[i] = [nr, nc]
bad = [i for i in range(k) if tuple(pos[i]) != targets[i]]
if bad:
    print(f"NOT AT TARGET: tokens {bad[:10]} (of {len(bad)}). e.g. token {bad[0]}: at {tuple(pos[bad[0]])} want {targets[bad[0]]}")
else:
    print(f"FEASIBLE: {L} actions, all {k} tokens at target")
