#!/usr/bin/env python3
# Show which numbers in [L,R] qualify per brute force.
import sys
def ds(x):
    return sum(int(c) for c in str(x))
def nl(x):
    return len(str(x))
L, R = int(sys.argv[1]), int(sys.argv[2])
good = []
for x in range(L, R+1):
    if x > 0 and ds(x) % nl(x) == 0:
        good.append((x, ds(x), nl(x)))
for x, s, l in good:
    print(f"x={x} digitsum={s} len={l} sum%len={s%l}")
print("count:", len(good))
