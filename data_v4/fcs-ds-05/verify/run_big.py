#!/usr/bin/env python3
import subprocess, sys, random
from collections import deque, defaultdict

SOL = "/tmp/fcs-ds-05_x"

def brute(data_str):
    data = data_str.split(); idx=0
    n=int(data[idx]); idx+=1; q=int(data[idx]); idx+=1
    edges=set(); out=[]
    for _ in range(q):
        typ=int(data[idx]); idx+=1; u=int(data[idx]); idx+=1; v=int(data[idx]); idx+=1
        if typ==1: edges.add((min(u,v),max(u,v)))
        elif typ==2: edges.discard((min(u,v),max(u,v)))
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

def gen_case(seed, nmax, qmax):
    rng=random.Random(seed)
    n=rng.randint(2,nmax); q=rng.randint(1,qmax)
    present=set(); lines=[]
    for _ in range(q):
        choices=["query","query"]
        maxe=n*(n-1)//2
        if len(present)<maxe: choices+=["add","add","add"]
        if present: choices+=["remove"]
        action=rng.choice(choices)
        if action=="query":
            lines.append(f"3 {rng.randint(1,n)} {rng.randint(1,n)}")
        elif action=="add":
            while True:
                u=rng.randint(1,n); v=rng.randint(1,n)
                if u==v: continue
                e=(min(u,v),max(u,v))
                if e not in present: break
            present.add(e); lines.append(f"1 {e[0]} {e[1]}")
        else:
            e=rng.choice(list(present)); present.discard(e); lines.append(f"2 {e[0]} {e[1]}")
    return f"{n} {q}\n" + "\n".join(lines) + "\n"

def run_sol(inp):
    return subprocess.run([SOL], input=inp, capture_output=True, text=True).stdout

N=int(sys.argv[1]); nmax=int(sys.argv[2]); qmax=int(sys.argv[3])
mism=0
for seed in range(1,N+1):
    inp=gen_case(seed*7+13, nmax, qmax)
    exp=brute(inp); got=run_sol(inp)
    if got!=exp:
        mism+=1
        if mism<=3:
            print("MISMATCH seed",seed); print("IN:\n"+inp); print("SOL:\n"+got); print("BR:\n"+exp)
print(f"cases={N} nmax={nmax} qmax={qmax} mismatches={mism}")
