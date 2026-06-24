import sys

def main():
    data = sys.stdin.read().split()
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

    # Independent brute: enumerate all leaves, and for each leaf walk up to root,
    # adding that leaf's "unit" to every edge on the path. Equivalently the total
    # pumping work = sum over leaves L of (sum of edge costs on root->L path).
    leaves = [i for i in range(1, n + 1) if len(children[i]) == 0]
    total = 0
    for L in leaves:
        v = L
        while v != root:
            total += cost[v]
            v = par[v]
    print(total)

main()
