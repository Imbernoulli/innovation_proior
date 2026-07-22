# TIER: invalid
# Emits an eviction directive at op 0 -- a fresh cache with a free slot needs no
# eviction, so this is a stray directive that must be rejected (Ratio 0).
import sys
sys.stdin.read()
sys.stdout.write("1\nEVICT 0 0\n")
