# TIER: greedy
# Textbook multiplicative hashing with a power-of-two hash-table size (the
# Knuth/Fibonacci-hashing idiom: h(x) = ((a*x) mod 2^k) mod B, picked once and
# reused for every input because "a big odd multiplier + a power-of-two modulus
# gives good spread for arbitrary keys"). This is exactly what an average
# strong coder pastes from memory without ever looking at the actual tracking
# codes. It is a fine choice for adversarial/uniform keys, but every code in
# these instances secretly lies on an affine lattice whose batch stride (the
# pairwise-difference GCD) is frequently a large power of two -- and a
# power-of-two hash modulus shares that entire factor, collapsing the whole
# lattice into a handful of residues before the final mod B.
import sys, json

inst = json.load(sys.stdin)
M = 1 << 24                 # "fast bitmask" table size -- ignores the data
a = 2654435769 % M          # Knuth's fixed multiplicative constant
if a % 2 == 0:
    a += 1                  # keep it odd (classic folklore for pow2 tables)
c = 12345 % M
print(json.dumps({"a": a, "c": c, "M": M}))
