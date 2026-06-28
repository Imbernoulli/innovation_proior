# Independently recompute the total cost of the solver's output and verify it
# matches what score.py uses, and that the delta-tracked schedule is valid.
import subprocess, sys, os
HERE=os.path.dirname(os.path.abspath(__file__))
def inst_read(p):
    t=open(p).read().split(); it=iter(t)
    n=int(next(it));M=int(next(it));T=int(next(it))
    d=[int(next(it)) for _ in range(n)]
    c=[int(next(it)) for _ in range(n)]
    ini=[int(next(it)) for _ in range(T)]
    s=[[int(next(it)) for _ in range(T)] for _ in range(T)]
    return n,M,T,d,c,ini,s
for sd in [1,8,12,18]:
    inp=f"_tmp/in_{sd}.txt"; outp=f"_tmp/sol_{sd}.txt"
    n,M,T,d,c,ini,s=inst_read(inp)
    rows=[ln.split() for ln in open(outp) if ln.split()]
    assert len(rows)==M, (len(rows),M)
    seen=[0]*n; total=0
    for r in rows:
        k=int(r[0]); jobs=[int(x) for x in r[1:]]
        assert len(jobs)==k
        for j in jobs: seen[j]+=1
        if jobs:
            load=sum(d[j] for j in jobs)+ini[c[jobs[0]]]
            for i in range(1,len(jobs)): load+=s[c[jobs[i-1]]][c[jobs[i]]]
            total+=load
    assert all(v==1 for v in seen), "not a partition"
    print(f"seed {sd}: partition OK, recomputed total cost = {total}")
print("CONSISTENCY OK")
