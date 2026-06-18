# 2-SAT via strongly connected components

## Problem

Given $n$ boolean variables $x_1, \dots, x_n$ and $m$ clauses, each a disjunction of two literals $(a \lor b)$ (a literal is a variable or its negation), decide whether all clauses can be satisfied simultaneously, and if so output a satisfying assignment. Sizes up to $n, m \approx 10^6$, so the solution must run in $O(n + m)$.

## Key idea

**A two-literal clause is a pair of implications.** $(a \lor b)$ is false only when both literals are false, so it is logically equivalent to

$$\lnot a \Rightarrow b \quad\text{and}\quad \lnot b \Rightarrow a.$$

Build the **implication graph** on $2n$ vertices, one per literal ($x$ and $\lnot x$ for each variable). Each clause $(a \lor b)$ contributes the two edges $\lnot a \to b$ and $\lnot b \to a$. A directed path means forced truth: if the literal at the start is true, every literal reachable from it is forced true. By construction the edge set is closed under "reverse and negate": $u \to v$ present $\Rightarrow$ $\lnot v \to \lnot u$ present (the contrapositive).

**Satisfiable iff no variable's two literals share a strongly connected component.** If $x$ reaches $\lnot x$ and $\lnot x$ reaches $x$ — i.e. $x$ and $\lnot x$ are in the same SCC — then either value of $x$ forces its negation, a contradiction, so the formula is unsatisfiable. This condition (some $x$ with $\mathrm{comp}[x] = \mathrm{comp}[\lnot x]$) is also sufficient; the assignment below is a witness.

**Assignment by topological order.** Condense the SCCs into a DAG and take a topological order; for each variable set the literal whose component is **later** in topological order (closer to a sink) to true. Equivalently $x$ is true iff $\mathrm{comp}[x]$ is later than $\mathrm{comp}[\lnot x]$.

*Why it is consistent.* The selected true literals are closed under implications. If a selected true literal $p$ reaches $q$, then the reverse-and-negate symmetry gives a mirrored path $\lnot q \rightsquigarrow \lnot p$. Since $p$ was selected true, $\lnot p$'s component is earlier than $p$'s. The paths give the topological inequalities

$$
\mathrm{comp}[\lnot q] \le \mathrm{comp}[\lnot p] < \mathrm{comp}[p] \le \mathrm{comp}[q],
$$

so $q$ is later than $\lnot q$ and is also selected true. Thus every implication out of a true literal is honored. In particular, a selected literal cannot force its own negation, and if $a$ is false in a clause $(a \lor b)$ then $\lnot a$ is true and the implication $\lnot a \Rightarrow b$ forces $b$ true. Hence every clause holds. This construction succeeds whenever no variable's two literals share a component, which is exactly the sufficiency claim.

**Sign of the comparison.** A single-pass SCC routine (depth-first search with discovery indices and low-links) numbers components in *reverse* topological order: a sink component finishes first and gets the smallest id, so a smaller id sits later in topological order. With literals encoded as vertices $2i$ (positive) and $2i+1$ (negated), the assignment is therefore $\mathrm{assign}[i] = \big(\mathrm{comp}[2i] < \mathrm{comp}[2i+1]\big)$, and unsatisfiability is $\mathrm{comp}[2i] = \mathrm{comp}[2i+1]$. (A two-pass routine whose final numbering runs in *forward* topological order would use $>$ instead — same principle, different id convention.)

## Algorithm

1. Encode literals: variable $i$ positive $\to 2i$, negated $\to 2i+1$; negation is `node ^ 1`.
2. For each clause $(a \lor b)$ add edges $\mathrm{neg}(a) \to b$ and $\mathrm{neg}(b) \to a$.
3. Compute SCCs of the $2n$-vertex graph in one iterative pass (ids in reverse topological order).
4. If any variable has $\mathrm{comp}[2i] = \mathrm{comp}[2i+1]$, report unsatisfiable.
5. Otherwise set $x_i$ true iff $\mathrm{comp}[2i] < \mathrm{comp}[2i+1]$.

Total $O(n + m)$ time and memory. The depth-first search must be iterative — a recursive one overflows the call stack at $2 \times 10^6$ vertices.

## Code

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
    total = 2 * n
    adj = [[] for _ in range(total)]
    for a, b in clauses:
        # (a OR b) is the two forced cases: not-a -> b and not-b -> a.
        adj[a ^ 1].append(b)
        adj[b ^ 1].append(a)
    comp = strongly_connected_components(total, adj)
    assign = []
    for i in range(n):
        positive = 2 * i
        negative = positive + 1
        if comp[positive] == comp[negative]:
            return None
        # This SCC routine closes sinkward components first, so smaller id is later.
        assign.append(comp[positive] < comp[negative])
    return assign


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

## Complexity

- **Time:** $O(n + m)$ — building the $2n$-vertex, $2m$-edge implication graph is $O(n + m)$, one SCC pass is $O(V + E) = O(n + m)$, the assignment scan is $O(n)$.
- **Memory:** $O(n + m)$ for the adjacency lists and the per-vertex arrays.
- The SCC pass is iterative so that an implication chain of length $\sim 2 \times 10^6$ does not overflow the recursion stack.
