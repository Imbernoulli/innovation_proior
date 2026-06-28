#!/usr/bin/env python3
"""
Random small-case generator for offline dynamic connectivity.

Usage: gen.py <seed>

Produces a valid operation stream:
  line 1: n q
  next q lines: "type u v"
    type 1 = add edge u-v   (only emitted if that edge is currently absent and u!=v)
    type 2 = remove edge u-v (only emitted if that edge is currently present)
    type 3 = query u-v       (any u, v in 1..n)

The generator maintains the live edge set so it never adds a duplicate edge or
removes a missing one -- matching the problem guarantee.
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 6)
    q = rng.randint(0, 14)

    present = set()
    lines = []

    for _ in range(q):
        # choose an action among those currently legal
        choices = ["query"]
        # we can add if there exists at least one absent edge with u != v
        max_edges = n * (n - 1) // 2
        if n >= 2 and len(present) < max_edges:
            choices.append("add")
        if present:
            choices.append("remove")

        action = rng.choice(choices)

        if action == "query":
            u = rng.randint(1, n)
            v = rng.randint(1, n)
            lines.append(f"3 {u} {v}")
        elif action == "add":
            # pick a random absent edge
            while True:
                u = rng.randint(1, n)
                v = rng.randint(1, n)
                if u == v:
                    continue
                e = (min(u, v), max(u, v))
                if e not in present:
                    break
            present.add(e)
            lines.append(f"1 {e[0]} {e[1]}")
        else:  # remove
            e = rng.choice(list(present))
            present.discard(e)
            lines.append(f"2 {e[0]} {e[1]}")

    out = [f"{n} {q}"]
    out.extend(lines)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
