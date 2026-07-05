#!/usr/bin/env python3
# gen.py <testId> : print ONE instance to stdout.
# testId 1..10 = difficulty ladder (small near-square -> large, tall, tight rmax).
# Deterministic: everything is a closed-form function of testId only.
import sys

def instance(t):
    # medium scale: N grows 12 -> 30
    N = 12 + 2 * (t - 1)
    # container: unit-ish width, height grows -> more extreme aspect ratio
    W = 1.0
    H = round(1.0 + 0.10 * (t - 1), 4)          # 1.0 .. 1.9
    # pipe stock cap: shrinks with difficulty so the cap starts to bind
    rmax = round(0.30 - 0.014 * (t - 1), 4)     # 0.300 .. 0.174
    return N, W, H, rmax

def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    if t > 10:
        t = 10
    N, W, H, rmax = instance(t)
    print("%d %s %s %s" % (N, repr(W), repr(H), repr(rmax)))

if __name__ == "__main__":
    main()
