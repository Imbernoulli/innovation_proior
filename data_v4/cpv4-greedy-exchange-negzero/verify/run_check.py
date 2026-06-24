import subprocess, sys, importlib.util, io

# In-process stress harness: import gen + brute as modules, pipe each generated
# input straight into the compiled sol via subprocess. No shared temp files, so
# concurrent agents cannot clobber anything.

HERE = "/srv/home/bohanlyu/innovation_proior/data_v4/cpv4-greedy-exchange-negzero/verify"
SOL = "/tmp/cpv4-greedy-exchange-negzero_sol"

# load gen.py and brute.py as modules
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

import random

def gen_input(seed):
    rng = random.Random(seed)
    r = rng.random()
    if r < 0.08:
        n = 0
    else:
        n = rng.randint(1, 9)
    flavor = rng.randint(0, 4)
    d = []; v = []
    for _ in range(n):
        d.append(rng.randint(0, n + 2))
        if flavor == 0:
            v.append(rng.randint(-9, 0))
        elif flavor == 1:
            v.append(0)
        elif flavor == 2:
            v.append(rng.randint(-9, 9))
        elif flavor == 3:
            v.append(rng.choice([rng.randint(-3, 0), rng.randint(1, 20)]))
        else:
            v.append(rng.randint(-2, 3))
    out = [str(n), " ".join(map(str, d)), " ".join(map(str, v))]
    return "\n".join(out) + "\n", n, d, v

def brute(d, v):
    n = len(d)
    best = 0
    for mask in range(1 << n):
        deadlines = []; total = 0
        for i in range(n):
            if mask & (1 << i):
                deadlines.append(d[i]); total += v[i]
        deadlines.sort()
        feasible = True
        for j, dl in enumerate(deadlines, start=1):
            if dl < j:
                feasible = False; break
        if feasible and total > best:
            best = total
    return best

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    mism = 0
    for seed in range(1, N + 1):
        inp, n, d, v = gen_input(seed)
        exp = brute(d, v)
        got = subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout.strip()
        if str(exp) != got:
            mism += 1
            if mism <= 5:
                print(f"MISMATCH seed={seed} sol=[{got}] brute=[{exp}]")
                print("INPUT:", repr(inp))
    print(f"TOTAL={N} MISMATCHES={mism}")

if __name__ == "__main__":
    main()
