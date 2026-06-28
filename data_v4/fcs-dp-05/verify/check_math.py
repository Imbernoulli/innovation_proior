import math
from itertools import permutations

def count_linear_extensions_bruteforce(n, parent):
    # parent[v] = parent of v in rooted tree, parent[root] = -1
    # count permutations (label assignment) where parent label < child label
    # i.e. number of topological orderings consistent with "parent before child"
    # Actually: count assignments of distinct labels 1..n such that each vertex < all descendants
    # equivalently number of linear extensions of the poset parent<child
    cnt = 0
    for perm in permutations(range(1, n+1)):
        # perm[v] = label of vertex v
        ok = True
        for v in range(n):
            p = parent[v]
            if p != -1 and perm[p] > perm[v]:
                ok = False
                break
        if ok:
            cnt += 1
    return cnt

def hook_formula(n, parent, root, children):
    # compute subtree sizes
    size = [0]*n
    order = []
    # iterative dfs
    visited=[False]*n
    stack=[root]
    visited[root]=True
    while stack:
        u=stack.pop()
        order.append(u)
        for c in children[u]:
            if not visited[c]:
                visited[c]=True
                stack.append(c)
    for u in reversed(order):
        size[u]=1
        for c in children[u]:
            size[u]+=size[c]
    prod=1
    for v in range(n):
        prod*=size[v]
    return math.factorial(n)//prod

# test on a small tree: rooted at each vertex
# tree edges
import sys
edges=[(0,1),(1,2),(1,3)]
n=4
adj={i:[] for i in range(n)}
for a,b in edges:
    adj[a].append(b); adj[b].append(a)

for root in range(n):
    # build parent/children
    parent=[-1]*n
    children={i:[] for i in range(n)}
    visited=[False]*n
    stack=[root]; visited[root]=True
    while stack:
        u=stack.pop()
        for w in adj[u]:
            if not visited[w]:
                visited[w]=True
                parent[w]=u
                children[u].append(w)
                stack.append(w)
    bf=count_linear_extensions_bruteforce(n,parent)
    hf=hook_formula(n,parent,root,children)
    print(f"root={root} brute={bf} hook={hf} match={bf==hf}")
