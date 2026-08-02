#!/usr/bin/env python3
"""Checker for fsx_C_1259. CLI: python3 counter.py <in> <out> <ans>
Prints '... Ratio: <float in [0,1]>' on its own final line and exits 0.
"""
import sys


def ceil_div(a, b):
    return -(-a // b)


def read_input(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)

    def nxt():
        return next(it)

    n = int(nxt()); m = int(nxt()); g = int(nxt())
    H_rate = int(nxt()); N_rate = int(nxt()); L_fetch = int(nxt()); L_edge = int(nxt())
    group_of = [int(nxt()) for _ in range(n)]
    flops = [0] * n
    mem_bytes = [0] * n
    for i in range(n):
        flops[i] = int(nxt())
        mem_bytes[i] = int(nxt())
    preds = [[] for _ in range(n)]
    for _ in range(m):
        u = int(nxt()); v = int(nxt()); b = int(nxt())
        preds[v].append((u, b))
    return dict(n=n, m=m, g=g, H_rate=H_rate, N_rate=N_rate, L_fetch=L_fetch,
                L_edge=L_edge, group_of=group_of, flops=flops, mem_bytes=mem_bytes,
                preds=preds)


def simulate(inst, device):
    """Deterministic two-serial-device makespan simulation. Kernels are processed
    in id order per device (edges only go low id -> high id, so id order is a
    valid topological order and needs no extra scheduling decision)."""
    n = inst['n']; flops = inst['flops']; mem_bytes = inst['mem_bytes']
    preds = inst['preds']; H_rate = inst['H_rate']; N_rate = inst['N_rate']
    L_fetch = inst['L_fetch']; L_edge = inst['L_edge']
    finish = [0] * n
    dev_free = [0, 0]  # 0 = host, 1 = near-memory
    makespan = 0
    for v in range(n):
        ready = 0
        for (u, b) in preds[v]:
            t = finish[u]
            if device[u] != device[v]:
                t += b * L_edge
            if t > ready:
                ready = t
        dv = device[v]
        rate = H_rate if dv == 0 else N_rate
        proc = ceil_div(flops[v], rate)
        if dv == 0:
            proc += mem_bytes[v] * L_fetch  # host must fetch its home-memory bytes
        start = ready if ready > dev_free[dv] else dev_free[dv]
        fin = start + proc
        finish[v] = fin
        dev_free[dv] = fin
        if fin > makespan:
            makespan = fin
    return makespan


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0  # bad invocation")
        return 0
    inp, outp = sys.argv[1], sys.argv[2]
    try:
        inst = read_input(inp)
        n, g = inst['n'], inst['g']

        base_device = [0] * n
        B = simulate(inst, base_device)  # trivial feasible baseline: all-host

        try:
            raw = open(outp).read()
        except Exception:
            print("Ratio: 0.0  # cannot read output file")
            return 0
        toks = raw.split()
        if len(toks) != g:
            print(f"Ratio: 0.0  # expected {g} tokens, got {len(toks)}")
            return 0

        assign = []
        for t in toks:
            try:
                v = int(t)
            except ValueError:
                print(f"Ratio: 0.0  # non-integer token {t!r}")
                return 0
            if v not in (0, 1):
                print(f"Ratio: 0.0  # token {v} not in {{0,1}}")
                return 0
            assign.append(v)

        device = [assign[inst['group_of'][i]] for i in range(n)]
        F = simulate(inst, device)
        if not (F > 0):
            print("Ratio: 0.0  # non-positive makespan")
            return 0

        sc = min(1000.0, 100.0 * B / max(1e-9, F))
        ratio = sc / 1000.0
        print(f"B={B} F={F} Ratio: {ratio:.6f}")
        return 0
    except Exception as e:
        print(f"Ratio: 0.0  # checker exception {e}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
