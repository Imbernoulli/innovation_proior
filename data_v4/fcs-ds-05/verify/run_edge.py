#!/usr/bin/env python3
import subprocess
from collections import deque, defaultdict

SOL = "/tmp/fcs-ds-05_x"

def brute(data_str):
    data = data_str.split()
    idx = 0
    n = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1
    edges = set(); out = []
    for _ in range(q):
        typ = int(data[idx]); idx += 1
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        if typ == 1: edges.add((min(u,v),max(u,v)))
        elif typ == 2: edges.discard((min(u,v),max(u,v)))
        else:
            if u==v: out.append("YES"); continue
            adj=defaultdict(list)
            for (a,b) in edges: adj[a].append(b); adj[b].append(a)
            seen={u}; dq=deque([u]); found=False
            while dq:
                x=dq.popleft()
                if x==v: found=True; break
                for y in adj[x]:
                    if y not in seen: seen.add(y); dq.append(y)
            out.append("YES" if found else "NO")
    return "\n".join(out)+("\n" if out else "")

def run_sol(inp):
    return subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout

cases = {
 "empty q=0": "1 0\n",
 "single vertex self-query": "1 1\n3 1 1\n",
 "two vertices no edge": "2 1\n3 1 2\n",
 "add then query": "2 2\n1 1 2\n3 1 2\n",
 "add remove query": "2 3\n1 1 2\n2 1 2\n3 1 2\n",
 "query before any add": "3 4\n3 1 3\n1 1 2\n1 2 3\n3 1 3\n",
 "re-add after remove": "3 6\n1 1 2\n2 1 2\n3 1 3\n1 2 3\n1 1 2\n3 1 3\n",
 "self query connected": "3 1\n3 2 2\n",
 "edge alive to end": "2 2\n1 1 2\n3 1 2\n",
 "diamond": "4 6\n1 1 2\n1 1 3\n1 2 4\n1 3 4\n2 2 4\n3 1 4\n",
}

bad=0
for name, inp in cases.items():
    got=run_sol(inp); exp=brute(inp)
    ok = (got==exp)
    if not ok:
        bad+=1
        print("FAIL", name); print("IN:",repr(inp)); print("GOT:",repr(got)); print("EXP:",repr(exp))
    else:
        print("ok  ", name)
print("edge bad =", bad)
