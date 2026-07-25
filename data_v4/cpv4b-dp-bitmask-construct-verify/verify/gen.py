#!/usr/bin/env python3
# Random SMALL-case generator: python3 gen.py <seed>
# Emits:  n L m  then m blacklisted length-L binary strings.
import sys, random

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(seed)

L = random.randint(1, 5)
n = random.randint(0, 16)
universe = 1 << L
# pick m forbidden patterns; sometimes many (to provoke -1), sometimes few
m_max = universe
m = random.randint(0, m_max)
pats = random.sample(range(universe), m)
forb = [format(p, '0' + str(L) + 'b') for p in pats]

print(n, L, m)
for f in forb:
    print(f)
