import subprocess, sys, random
HERE = "/srv/home/bohanlyu/innovation_proior/data_v4/cpv4-sorting-sweep-greedytrap/verify"
SOL = "/tmp/cpv4-sorting-sweep-greedytrap_sol"

def gen(seed):
    rng = random.Random(seed)
    n = rng.randint(0, 12)
    T = rng.choice([4, 6, 8, 10])
    lines = [str(n)]
    for _ in range(n):
        a = rng.randint(0, T); b = rng.randint(0, T)
        if a == b: b = a + 1
        s, f = min(a, b), max(a, b)
        p = rng.randint(1, 20)
        lines.append(f"{s} {f} {p}")
    return "\n".join(lines) + "\n"

def brute(inp):
    out = subprocess.run([sys.executable, HERE + "/brute.py"], input=inp,
                         capture_output=True, text=True)
    return out.stdout.strip()

def sol(inp):
    out = subprocess.run([SOL], input=inp, capture_output=True, text=True)
    return out.stdout.strip()

mism = 0; total = 0
seeds = list(range(1, 501)) + list(range(2000, 2101))
for seed in seeds:
    inp = gen(seed)
    a = sol(inp); b = brute(inp)
    total += 1
    if a != b:
        mism += 1
        if mism <= 3:
            print(f"MISMATCH seed={seed} sol={a} bru={b}")
            print(inp)
print(f"Total: {mism} mismatches / {total} cases")
