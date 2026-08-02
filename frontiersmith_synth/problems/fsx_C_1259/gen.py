#!/usr/bin/env python3
"""Generator for fsx_C_1259 -- near-memory-offload kernel-graph placement.

python3 gen.py <testId>   -> prints one instance to stdout. Seeded by testId only.

Instance =
  n kernels forming a DAG (edges u->v, u<v so kernel id order is a valid
  topological order), each with a flops cost and a "home-memory read" byte
  size; a partition of the kernels into contiguous offload-granularity
  GROUPS (the unit a solver actually assigns); a Host device and a
  Near-Memory (NM) device with distinct throughputs; a per-byte cost for
  pulling data from main memory into the Host (fetch cost) and a per-byte
  cost for moving an activation across the Host<->NM boundary along a
  dependency edge.

Each GROUP is planted with one of three shapes (kernels inside a group share
the shape, so the group's own aggregate flops/bytes cleanly reflects it):
  - compute-dense : big flops, tiny bytes    -> always wants Host
  - data-heavy    : tiny flops, big bytes    -> always wants NM
  - mixed-trap    : BIG flops *and* big bytes -> wants Host whenever the
                     arithmetic-intensity (flops/bytes) clears the
                     rate-derived threshold, even though its byte volume
                     alone looks identical to a data-heavy group.
mixed-trap groups are only planted on "trap" testIds, where the near-memory
device is deliberately slow (large H_rate/N_rate gap) -- exactly the regime
where a rule that offloads purely by byte volume is wrong.
"""
import sys
import random


def main():
    testId = int(sys.argv[1])
    rng = random.Random(10_000 + 97 * testId)

    sizes = [8, 10, 14, 18, 22, 28, 34, 40, 46, 54]
    n = sizes[min(testId, len(sizes)) - 1]

    H_rate = 100
    trap = testId >= 3          # 8 of the 10 cases are trap cases (>=3 required)
    if trap:
        N_rate = rng.choice([10, 11, 12])       # near-memory is weak: big asymmetry
    else:
        N_rate = rng.choice([40, 50, 60])       # near-memory is fairly competitive
    L_fetch = 1                                  # host must pay this per home-memory byte
    L_edge = rng.choice([1, 1, 2])               # cross-device edge cost per byte

    # ---- offload-granularity groups: contiguous partition of [0,n), each
    #      group tagged with a planted shape shared by all its kernels ----
    trap_frac = 0.22 if trap else 0.0
    data_frac = 0.45

    group_of = []
    group_shapes = []
    g = 0
    i = 0
    while i < n:
        size = rng.choices([1, 1, 1, 2, 2, 3], k=1)[0]
        size = min(size, n - i)
        r = rng.random()
        if trap and r < trap_frac:
            shape = "trap"
        elif r < trap_frac + data_frac:
            shape = "data"
        else:
            shape = "compute"
        group_of.extend([g] * size)
        group_shapes.append(shape)
        i += size
        g += 1

    flops = [0] * n
    mem_bytes = [0] * n
    for i in range(n):
        shape = group_shapes[group_of[i]]
        if shape == "trap":
            # mixed-trap: big flops AND big bytes, intensity clears the trap tau
            flops[i] = rng.randint(7500, 9500)
            mem_bytes[i] = rng.randint(280, 420)
        elif shape == "data":
            # data-heavy: tiny flops, big bytes
            flops[i] = rng.randint(10, 60)
            mem_bytes[i] = rng.randint(180, 440)
        else:
            # compute-dense: big flops, tiny bytes
            flops[i] = rng.randint(1500, 3000)
            mem_bytes[i] = rng.randint(3, 8)

    # ---- dependency DAG: 1-2 predecessors from a nearby window, u < v ----
    edges = []
    for v in range(1, n):
        lo = max(0, v - 4)
        window = v - lo
        k = 1 if window < 2 or rng.random() < 0.5 else 2
        k = min(k, window)
        preds = rng.sample(range(lo, v), k)
        for u in preds:
            b = rng.randint(50, 400)
            edges.append((u, v, b))
    m = len(edges)

    out = [f"{n} {m} {g} {H_rate} {N_rate} {L_fetch} {L_edge}",
           " ".join(map(str, group_of))]
    for i in range(n):
        out.append(f"{flops[i]} {mem_bytes[i]}")
    for (u, v, b) in edges:
        out.append(f"{u} {v} {b}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
