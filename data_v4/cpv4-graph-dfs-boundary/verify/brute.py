import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    if n <= 0:
        print(0)
        return
    par = [0] * n
    p = [0] * n
    for i in range(n):
        pa = int(data[idx]); idx += 1
        pw = int(data[idx]); idx += 1
        par[i] = pa
        p[i] = pw

    # depth of each node by walking up to the root
    def depth(v):
        d = 0
        while par[v] != -1:
            v = par[v]
            d += 1
        return d

    dep = [depth(i) for i in range(n)]

    covered = 0
    for v in range(n):
        dv = dep[v]
        ok = False
        # walk strictly upward: u ranges over PROPER ancestors of v only
        u = par[v]
        while u != -1:
            if dv - dep[u] <= p[u]:   # station u reaches v: distance <= p[u]
                ok = True
                break
            u = par[u]
        if ok:
            covered += 1
    print(covered)

main()
