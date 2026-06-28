#!/usr/bin/env python3
import subprocess, sys, importlib.util, io, random
from collections import deque, defaultdict

SOL = "/tmp/fcs-ds-05_x"

def brute(data_str):
    data = data_str.split()
    idx = 0
    n = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1
    edges = set()
    out = []
    for _ in range(q):
        typ = int(data[idx]); idx += 1
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        if typ == 1:
            edges.add((min(u, v), max(u, v)))
        elif typ == 2:
            edges.discard((min(u, v), max(u, v)))
        else:
            if u == v:
                out.append("YES"); continue
            adj = defaultdict(list)
            for (a, b) in edges:
                adj[a].append(b); adj[b].append(a)
            seen = {u}; dq = deque([u]); found = False
            while dq:
                x = dq.popleft()
                if x == v: found = True; break
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y); dq.append(y)
            out.append("YES" if found else "NO")
    return "\n".join(out) + ("\n" if out else "")

# import gen
spec = importlib.util.spec_from_file_location("gen", "/srv/home/bohanlyu/innovation_proior/data_v4/fcs-ds-05/verify/gen.py")

def gen_case(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 6)
    q = rng.randint(0, 14)
    present = set()
    lines = []
    for _ in range(q):
        choices = ["query"]
        max_edges = n * (n - 1) // 2
        if n >= 2 and len(present) < max_edges:
            choices.append("add")
        if present:
            choices.append("remove")
        action = rng.choice(choices)
        if action == "query":
            u = rng.randint(1, n); v = rng.randint(1, n)
            lines.append(f"3 {u} {v}")
        elif action == "add":
            while True:
                u = rng.randint(1, n); v = rng.randint(1, n)
                if u == v: continue
                e = (min(u, v), max(u, v))
                if e not in present: break
            present.add(e); lines.append(f"1 {e[0]} {e[1]}")
        else:
            e = rng.choice(list(present)); present.discard(e)
            lines.append(f"2 {e[0]} {e[1]}")
    out = [f"{n} {q}"]; out.extend(lines)
    return "\n".join(out) + "\n"

def run_sol(inp):
    r = subprocess.run([SOL], input=inp, capture_output=True, text=True)
    return r.stdout

mism = 0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
for seed in range(1, N + 1):
    inp = gen_case(seed)
    exp = brute(inp)
    got = run_sol(inp)
    if got != exp:
        mism += 1
        if mism <= 3:
            print("MISMATCH seed", seed)
            print("INPUT:\n" + inp)
            print("SOL:\n" + got)
            print("BRUTE:\n" + exp)
print(f"cases={N} mismatches={mism}")
