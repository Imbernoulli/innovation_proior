import sys, subprocess, random, importlib.util, io

# Load gen and brute as modules by exec to avoid 800 subprocess spawns.
def load_src(path):
    with open(path) as f:
        return f.read()

# We'll reimplement gen and brute inline using their logic, but to keep them
# "the same" we instead call them via functions parsed from the files.

def gen_case(seed):
    random.seed(seed)
    n = random.randint(1, 8)
    out = [str(n)]
    for i in range(1, n + 1):
        if i == 1:
            p = -1
        else:
            p = random.randint(1, i - 1)
        c = random.randint(0, 6)
        out.append(f"{p} {c}")
    return "\n".join(out) + "\n"

def brute(text):
    data = text.split()
    idx = 0
    n = int(data[idx]); idx += 1
    par = [0] * (n + 1)
    cost = [0] * (n + 1)
    children = [[] for _ in range(n + 1)]
    root = -1
    for i in range(1, n + 1):
        p = int(data[idx]); idx += 1
        c = int(data[idx]); idx += 1
        par[i] = p
        cost[i] = c
        if p == -1 or p == 0:
            root = i
        else:
            children[p].append(i)
    leaves = [i for i in range(1, n + 1) if len(children[i]) == 0]
    total = 0
    for L in leaves:
        v = L
        while v != root:
            total += cost[v]
            v = par[v]
    return str(total)

SOL = "/tmp/cpv4-graph-dfs-overflow_sol"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
mism = 0
shown = 0
for s in range(1, N + 1):
    text = gen_case(s)
    exp = brute(text)
    got = subprocess.run([SOL], input=text, capture_output=True, text=True).stdout.strip()
    if got != exp:
        mism += 1
        if shown < 5:
            shown += 1
            print(f"MISMATCH seed={s} sol={got} bru={exp}")
            print("--- input ---")
            print(text)
print(f"CASES={N} TOTAL_MISMATCHES={mism}")
