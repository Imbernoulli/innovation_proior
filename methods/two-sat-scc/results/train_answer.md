I am handed $n$ boolean variables and $m$ clauses, each a disjunction of exactly two literals $(a \lor b)$ where a literal is a variable or its negation, and I must decide whether some assignment satisfies every clause at once and produce one when it exists. With $n, m$ up to about $10^6$, brute force over all $2^n$ assignments is gone before I start — even $2^{60}$ is hopeless. General SAT with arbitrary-width clauses is the canonical hard problem and offers no fast handle, so the only way through is to exploit the one structural fact I am given: every clause has exactly two literals. That specialness has to carry the entire solution, and the requirement is sharp — linear time, $O(n + m)$, with a witness assignment on output, not just a yes/no.

The method is 2-SAT via strongly connected components. Its starting observation is that a two-literal clause is not really a disjunction at all but a pair of implications. The clause $(a \lor b)$ is false in exactly one of the four cases — both literals false — and satisfied in the other three, so its entire content is "not both false." Turn that into forcing statements: if $a$ is false the clause survives only if $b$ is true, and symmetrically if $b$ is false then $a$ is true, which is exactly
$$\lnot a \Rightarrow b \quad\text{and}\quad \lnot b \Rightarrow a.$$
These two conditionals say "not both false," no more and no less — when $a$ is true the first premise is vacuous and the second is empty, when $a$ is false the first forces $b$ — so a clause and this pair of implications are interchangeable with zero loss. Implications chain, and chaining is what graphs are for, which is the lever.

So I build the implication graph on $2n$ vertices, one per literal: for each variable $x$ a vertex for "$x$ true" and a vertex for "$x$ false." Each clause $(a \lor b)$ contributes its two implications as two directed edges, $\lnot a \to b$ and $\lnot b \to a$, and because a literal like $\lnot a$ is itself one of the $2n$ vertices these are honest edges between existing nodes. A directed path now means a forced chain: commit the starting literal to true and every literal along the path is dragged true with it, so reachability is propagation of forced truth. One symmetry is baked into how the edges were added and it is load-bearing: whenever clause $(a \lor b)$ puts in $u \to v$ it also puts in $\lnot v \to \lnot u$ (with $u = \lnot a, v = b$ this is $\lnot b \to a$), so the edge set is closed under "reverse the arrow and negate both endpoints." That is just the contrapositive — $\lnot a \Rightarrow b$ is the same statement as $\lnot b \Rightarrow a$ — but as a structural property of the graph it is what makes the correctness proof close.

Now ask when the formula is hopeless. Suppose, following edges, $x$ can reach $\lnot x$: starting from "$x$ true" I am forced to "$x$ false," so committing $x$ true is self-defeating. If moreover $\lnot x$ can reach $x$, then committing $x$ false is equally self-defeating, and since the clauses are fixed truths I must honor, whichever value I try for $x$ drives me to its opposite — a flat contradiction. Two-way reachability is exactly the statement that $x$ and $\lnot x$ lie in the same strongly connected component, the maximal sets of mutually reachable vertices. So the formula is unsatisfiable precisely when some variable's two literal-vertices share a component, and this is checkable in linear time. The necessity is immediate; the claim I have to earn is sufficiency — that if no variable's two literals ever share a component, a satisfying assignment exists — and I want a construction that hands me the witness rather than a bare existence proof.

The construction comes from condensing the graph: collapse each strongly connected component to a super-node, which leaves a DAG because every cycle lived inside one component and got absorbed. A DAG has a topological order, and writing $\mathrm{comp}[v]$ for the position of $v$'s component in it, an edge $u \to v$ forces $u$'s component no later than $v$'s. For each variable the two literals sit in two different components (the assumption), so one is strictly later than the other in topological order, and the rule is: set the later literal — the one nearer a sink — to true. The reason it must be the later one and not an arbitrary choice is that truth propagates forward along edges, so declaring a downstream literal true forces only things even further downstream, where there is no room for the forced cone to swing back and contradict. To verify this is genuinely consistent and not just free of immediate self-contradiction, suppose a selected true literal $p$ reaches some $q$; then $p$'s component is no later than $q$'s, and the reverse-and-negate symmetry mirrors that path into $\lnot q \rightsquigarrow \lnot p$, giving $\lnot q$'s component no later than $\lnot p$'s. Since $p$ was chosen true, $\lnot p$ is strictly earlier than $p$, and chaining the order facts yields
$$
\mathrm{comp}[\lnot q] \le \mathrm{comp}[\lnot p] < \mathrm{comp}[p] \le \mathrm{comp}[q],
$$
so $q$ is the later literal of its own variable and is also set true. The selected literals are therefore closed under forcing: a chosen literal can never force its own negation, and if $a$ comes out false in a clause $(a \lor b)$ then $\lnot a$ is true and the honored implication $\lnot a \Rightarrow b$ makes $b$ true, so every clause holds. This succeeds exactly whenever no variable's literals share a component, which is the sufficiency I owed — settling the criterion both ways.

One concrete thing must be pinned down rather than guessed: which direction the component-finding routine numbers components, because "later in topological order is true" has to become an integer comparison and a flipped sign would silently produce plausible-looking wrong answers. I compute components in a single pass by depth-first search with discovery indices and low-links, popping a component the moment its root finishes. The closing order is significant — a sink component, having nowhere downstream to wait on, gets closed and numbered first — so this routine numbers components in reverse topological order: smaller id means closer to a sink means later in topological order. Encoding variable $i$'s positive literal at vertex $2i$ and its negation at $2i+1$, so that negation is the single operation $u \oplus 1$ on adjacent integers, the assignment becomes $\mathrm{assign}[i] = \big(\mathrm{comp}[2i] < \mathrm{comp}[2i+1]\big)$ and the unsatisfiability test is $\mathrm{comp}[2i] = \mathrm{comp}[2i+1]$. (A two-pass routine numbering in forward topological order would flip the comparison to $>$ — same principle, different id convention.) The last practical hazard at this scale is that a recursive depth-first search would blow the call stack on a long implication chain, so the search runs iteratively with an explicit stack of (vertex, adjacency-position) frames simulating the recursion: push a frame when recursing into an unvisited neighbor, do the low-link update against the parent and pop when neighbors are exhausted, and close a component whenever a vertex's low-link equals its own discovery index. Everything stays $O(n + m)$ in time and memory.

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
