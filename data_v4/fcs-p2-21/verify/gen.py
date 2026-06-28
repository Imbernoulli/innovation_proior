#!/usr/bin/env python3
"""
Random + edge-case test generator for the Box Stacking problem.

Usage: python3 gen.py <seed> [mode]
Prints a valid stdin instance to stdout.

Modes (auto-chosen from seed if not given):
  tiny     - n in 0..3, small dims, exercises corners (n=0, n=1, equal dims)
  small    - n in 0..6, dims 1..6  (good for differential testing vs brute)
  dup      - many duplicate/equal-dimension boxes (cubes, ties in area)
  greedy   - structured to try to fool volume/height greedy heuristics
  big      - n up to 200, dims up to 1e6 (stress; brute may be slow, used sparingly)
"""
import sys
import random


def emit(n, boxes):
    out = [str(n)]
    for (a, b, c) in boxes:
        out.append(f"{a} {b} {c}")
    sys.stdout.write("\n".join(out) + "\n")


def gen_tiny(rng):
    n = rng.randint(0, 3)
    boxes = []
    for _ in range(n):
        boxes.append((rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4)))
    return n, boxes


def gen_small(rng):
    n = rng.randint(0, 6)
    boxes = []
    for _ in range(n):
        boxes.append((rng.randint(1, 6), rng.randint(1, 6), rng.randint(1, 6)))
    return n, boxes


def gen_dup(rng):
    n = rng.randint(1, 7)
    boxes = []
    pool = []
    for _ in range(rng.randint(1, 3)):
        v = rng.randint(1, 5)
        pool.append((v, v, v))  # cubes -> lots of area ties
        pool.append((v, v, rng.randint(1, 5)))
    for _ in range(n):
        boxes.append(rng.choice(pool))
    return n, boxes


def gen_greedy(rng):
    # Build instances where a single huge-volume or huge-height box tempts a greedy
    # pick but blocks a better tall slim stack.
    n = rng.randint(2, 6)
    boxes = []
    # one fat short box (big volume, big base, small height)
    boxes.append((rng.randint(8, 10), rng.randint(8, 10), 1))
    # several slim tall stackable boxes
    base = rng.randint(2, 6)
    for _ in range(n - 1):
        w = rng.randint(1, base)
        d = rng.randint(1, base)
        boxes.append((w, d, rng.randint(5, 12)))
    return len(boxes), boxes


def gen_big(rng):
    n = rng.randint(150, 200)
    boxes = []
    for _ in range(n):
        boxes.append((rng.randint(1, 10**6), rng.randint(1, 10**6), rng.randint(1, 10**6)))
    return n, boxes


MODES = {
    "tiny": gen_tiny,
    "small": gen_small,
    "dup": gen_dup,
    "greedy": gen_greedy,
    "big": gen_big,
}


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    if len(sys.argv) > 2:
        mode = sys.argv[2]
    else:
        mode = rng.choice(["tiny", "small", "small", "dup", "greedy"])
    n, boxes = MODES[mode](rng)
    emit(n, boxes)


if __name__ == "__main__":
    main()
