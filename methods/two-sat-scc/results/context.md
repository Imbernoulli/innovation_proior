# Context

## Problem

Given n boolean variables x_1..x_n and m clauses, each of the form (a OR b) where a, b are literals (a variable or its negation), decide whether the formula is satisfiable, and if so output a satisfying assignment. (2-SAT; n, m up to ~10^6.)

## Code framework

```python
import sys


def strongly_connected_components(n, adj):
    """Iterative SCC. Returns comp[0..n-1], the component id of each vertex.
    Two vertices share an id exactly when each is reachable from the other.
    Component ids follow the order in which components are closed. O(V + E)."""
    index = [0] * n
    low = [0] * n
    on_stack = [False] * n
    comp = [-1] * n
    stack = []
    counter = 0
    cid = 0
    for s in range(n):
        if index[s]:
            continue
        work = [(s, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                counter += 1
                index[v] = low[v] = counter
                stack.append(v)
                on_stack[v] = True
            recursed = False
            while pi < len(adj[v]):
                u = adj[v][pi]
                pi += 1
                if index[u] == 0:
                    work[-1] = (v, pi)
                    work.append((u, 0))
                    recursed = True
                    break
                elif on_stack[u] and index[u] < low[v]:
                    low[v] = index[u]
            if recursed:
                continue
            if low[v] == index[v]:
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp[w] = cid
                    if w == v:
                        break
                cid += 1
            work.pop()
            if work:
                pv = work[-1][0]
                if low[v] < low[pv]:
                    low[pv] = low[v]
    return comp


def solve(n, clauses):
    """n boolean variables; clauses a list of (a, b), each literal encoded as
    an integer node: variable i positive is 2*i, negated is 2*i+1. Return None
    if unsatisfiable, else assign[0..n-1] of booleans."""
    # TODO
    pass


def lit_to_node(tok):
    v = int(tok)
    var = abs(v) - 1
    return 2 * var + (0 if v > 0 else 1)


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    clauses = []
    for _ in range(m):
        a = lit_to_node(next(it))
        b = lit_to_node(next(it))
        clauses.append((a, b))
    res = solve(n, clauses)
    out = []
    if res is None:
        out.append("UNSATISFIABLE")
    else:
        out.append("SATISFIABLE")
        out.append(" ".join(str(int(x)) for x in res))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
```
