import sys, random, subprocess, importlib.util

SOL = "/tmp/cpv4-segment-tree-boundary_sol"

# import the generator and brute as modules by exec, but they read stdin/argv,
# so instead reimplement inline to avoid process spawns for them.

def gen_input(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 12)
    q = rng.randint(1, 18)
    VMAX = rng.choice([1, 2, 3, 5, 9])
    lines = ["%d %d" % (n, q)]
    lines.append(" ".join(str(rng.randint(0, VMAX)) for _ in range(n)))
    ops = []
    for _ in range(q):
        if rng.randint(1, 2) == 1:
            p = rng.randint(1, n)
            x = rng.randint(0, VMAX)
            ops.append((1, p, x))
        else:
            l = rng.randint(1, n)
            r = rng.randint(l, n)
            ops.append((2, l, r))
        lines.append(" ".join(map(str, ops[-1])))
    return n, q, ops, "\n".join(lines) + "\n"

def brute(n, ops, init_vals):
    a = [0] + list(init_vals)     # 1-indexed
    out = []
    for op in ops:
        if op[0] == 1:
            _, p, x = op
            a[p] = x
        else:
            _, l, r = op
            best = 1
            cur = 1
            for i in range(l + 1, r + 1):
                if a[i] > a[i - 1]:
                    cur += 1
                else:
                    cur = 1
                if cur > best:
                    best = cur
            out.append(str(best))
    return "\n".join(out) + ("\n" if out else "")

def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    mism = 0
    shown = 0
    for seed in range(1, total + 1):
        rng = random.Random(seed)
        # regenerate to also capture init vals deterministically
        n = rng.randint(1, 12)
        q = rng.randint(1, 18)
        VMAX = rng.choice([1, 2, 3, 5, 9])
        init_vals = [rng.randint(0, VMAX) for _ in range(n)]
        ops = []
        lines = ["%d %d" % (n, q), " ".join(map(str, init_vals))]
        for _ in range(q):
            if rng.randint(1, 2) == 1:
                p = rng.randint(1, n); x = rng.randint(0, VMAX)
                ops.append((1, p, x))
            else:
                l = rng.randint(1, n); r = rng.randint(l, n)
                ops.append((2, l, r))
            lines.append(" ".join(map(str, ops[-1])))
        inp = "\n".join(lines) + "\n"

        got = subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout
        exp = brute(n, ops, init_vals)
        if got != exp:
            mism += 1
            if shown < 3:
                shown += 1
                print("=== MISMATCH seed", seed, "===")
                print(inp)
                print("--- sol ---"); print(got)
                print("--- brute ---"); print(exp)
    print("TOTAL_MISMATCHES=%d over %d cases" % (mism, total))

main()
