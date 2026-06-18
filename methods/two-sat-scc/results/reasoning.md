In front of me are $n$ boolean variables and a pile of $m$ constraints, each one a clause $(a \lor b)$ where $a$ and $b$ are literals — a variable or its negation. I have to decide whether there's any assignment of true/false to the variables making every clause hold at once, and produce one if it exists. With $n, m$ up to a million, the naive thing — try all $2^n$ assignments — is gone before I start; even $2^{60}$ is hopeless, let alone $2^{10^6}$. So I need structure in the fact that every clause has exactly *two* literals. General SAT with arbitrary-width clauses is the canonical hard problem, but two literals per clause is special, and I should lean on that specialness hard.

Let me stare at a single clause $(a \lor b)$ and ask what it actually forbids. It's false in exactly one situation: both $a$ and $b$ false. In the other three it's satisfied. So the content of the clause is purely "not both false." Now turn that around into a *forcing* statement. If $a$ happens to be false, then for the clause to survive, $b$ is forced true. And symmetrically, if $b$ is false then $a$ is forced true. That's it — the whole clause is captured by two conditionals:

$$\lnot a \Rightarrow b \quad\text{and}\quad \lnot b \Rightarrow a.$$

These two implications together say exactly "not both false," no more and no less. If $a$ is true the first implication's premise is vacuous and the second says nothing new; if $a$ is false the first forces $b$; same with the roles swapped. So a two-literal clause is, with no loss, a *pair of implications*. That feels like the lever — implications chain, and chaining is what graphs are for.

So let me make a graph whose vertices are *literals*. For every variable $x$ I'll have two vertices, one standing for "$x$ is true" and one for "$x$ is false," i.e. one for the literal $x$ and one for the literal $\lnot x$. There are $2n$ of these. Each clause contributes its two implications as two directed edges: $(a \lor b)$ gives an edge from $\lnot a$ to $b$ and an edge from $\lnot b$ to $a$. Here a literal like $\lnot a$ is itself one of my $2n$ vertices, so these are honest edges between existing vertices. A directed path in this graph then means a forced chain: if I commit to the literal at the start being true, every literal along the path is dragged true with it. Reachability is propagation of forced truth.

Let me notice a symmetry baked into how I added the edges, because I think it'll matter. Whenever I put in an edge $u \to v$ from clause $(a\lor b)$ — say $u = \lnot a$, $v = b$ — the *same clause* also put in $\lnot v \to \lnot u$, here $\lnot b \to a$. So the edge set is closed under "reverse the arrow and negate both endpoints": $u \to v$ present implies $\lnot v \to \lnot u$ present. That's just the contrapositive — $\lnot a \Rightarrow b$ is logically the same as $\lnot b \Rightarrow a$ — but having it as a structural property of the graph is going to be useful. Hold onto it.

Now, when is the whole thing unsatisfiable? Implications propagate, so think about what a path can do to a single variable $x$. Suppose, following edges, $x$ can reach $\lnot x$: starting from "$x$ true" I'm forced eventually to "$x$ false." So committing $x$ true is self-defeating. Fine — then maybe $x$ should be false. But suppose *also* $\lnot x$ can reach $x$: starting from "$x$ false" I'm forced to "$x$ true." Now both commitments are self-defeating. $x$ true forces $x$ false, and $x$ false forces $x$ true. There is no escape: whichever value I try for $x$, the implications drive me to its opposite, which (since the clauses are fixed truths I must honor) is a flat contradiction. So if $x$ reaches $\lnot x$ *and* $\lnot x$ reaches $x$, the formula is unsatisfiable.

That two-way reachability is a familiar object. Two vertices each reachable from the other lie in the same strongly connected component — the maximal sets of mutually-reachable vertices, the things you get by condensing every cycle. So my unsatisfiability condition is exactly: some variable $x$ has its two literal-vertices $x$ and $\lnot x$ in the *same* strongly connected component. If they're in the same component, $x$ and $\lnot x$ are mutually reachable, contradiction, dead. If for *every* variable $x$ the vertices $x$ and $\lnot x$ sit in *different* components, then this particular contradiction never fires. I can compute strongly connected components of a directed graph in linear time, so checking "no variable has both its literals in one component" is $O(n + m)$. The necessity is clear — same component really does kill it. The interesting claim, the one I have to earn, is that this is also *sufficient*: if no variable's two literals ever share a component, a satisfying assignment exists. And I'd better not just assert it — I want a construction that hands me the assignment, because the problem demands one.

So suppose every $x$ and $\lnot x$ are in different components, and let me try to *build* an assignment. Condense the graph: collapse each strongly connected component to a single super-node, keep an edge between super-nodes when there's an edge between their members. The condensation is a DAG — no cycles left, because every cycle lived inside one component and got collapsed. A DAG has a topological order: I can line the components up so that every edge points from an earlier component to a later one (or, reading the other way, every edge goes from a component toward the "downstream" end). Let me write $\mathrm{comp}[v]$ for the position of $v$'s component in this order. Inside one component all vertices share a $\mathrm{comp}$ value; across components, an edge $u \to v$ forces $u$'s component to come no later than $v$'s.

Now I have to assign a truth value to each variable, and the only handle I have is where its two literals' components fall in this order. For variable $x$ the two candidate literals are $x$ and $\lnot x$, living in two *different* components (that's the assumption), so one of them is strictly later in the topological order than the other. Intuition: a literal that's "downstream" — late in topological order, near a sink — has few or no implications leading *out* of it that could come back to bite me; a literal that's "upstream" near a source forces a long chain of consequences. Truth propagates *forward* along edges (if I set a literal true, everything it reaches must be true). So if I'm going to declare a literal true, I'd rather it be the downstream one: setting the late literal true forces only things even later, and there's less room for that forced cone to swing back and contradict me. Let me make that the rule and then check it actually never contradicts: for each variable, set the literal whose component comes *later* in topological order to true (and the other, earlier one, to false). Concretely, $x$ is true exactly when $\mathrm{comp}[x]$ is later than $\mathrm{comp}[\lnot x]$.

I need to verify this is stronger than just "no immediate self-contradiction." It has to respect every implication: whenever a literal I set true reaches another literal, that reached literal must also be set true. Say I've set literal $p$ true, meaning $p$'s component is later than $\lnot p$'s component. Now suppose there is a path $p \rightsquigarrow q$. The reverse-and-negate symmetry mirrors that path into $\lnot q \rightsquigarrow \lnot p$. In topological-order language, the first path says $p$'s component is no later than $q$'s, and the mirrored path says $\lnot q$'s component is no later than $\lnot p$'s. But $\lnot p$ is earlier than $p$, because $p$ was selected true. Chaining those order facts gives

$$
\mathrm{comp}[\lnot q] < \mathrm{comp}[q].
$$

So $q$ is also the later literal of its variable, hence $q$ is set true. Good — the selected literals are closed under forcing. As a special case, $p$ cannot reach $\lnot p$: closure would force $\lnot p$ true too, or directly the topological inequalities would require $p$'s later component to reach the earlier component of $\lnot p$. And if some chosen true literal reached both $y$ and $\lnot y$, closure would set both of them true, which would put a variable on both sides of the assignment; equivalently the two mirrored paths would stitch into a path from the chosen literal to its own negation. So the assignment is consistent and every implication it triggers is honored.

That closure argument is exactly why the rule has to pick the *later* literal and not an arbitrary one: the proof uses the strict order $\mathrm{comp}[\lnot p] < \mathrm{comp}[p]$ for the selected literal. And notice it proves the construction succeeds *whenever* no variable's two literals share a component, which is precisely the sufficiency I owed. So the criterion is settled both ways — satisfiable if and only if every variable has its two literals in different strongly connected components — and when satisfiable, "later component is true" reads off a witness. Every clause $(a \lor b)$ ends up satisfied because if $a$ came out false, then $\lnot a$ is true, the implication $\lnot a \Rightarrow b$ is honored by closure, and therefore $b$ is true; the symmetric case handles $b$ false.

Now to turn this into something that runs at a million variables. I need to be careful about one concrete thing: *which direction* does my component-finding routine number the components, because the rule "later in topological order is true" has to be translated into an actual integer comparison, and getting the sign backwards would quietly produce wrong assignments that still look plausible. Let me pin it down rather than guess. I'll compute strongly connected components by the single-pass method that uses a depth-first search with discovery indices and low-links, popping a component off a stack the moment a component's root finishes. The order in which that method *closes* components is significant: a component all of whose out-edges lead to already-finished components is finished first. A sink component — one with no edges leaving it in the condensation — has nowhere downstream to wait on, so it gets closed, and numbered, before its upstream predecessors. In other words this routine numbers components in *reverse* topological order: smaller id = closer to a sink = *later* in topological order. So "the literal whose component is later in topological order is true" becomes, in terms of the ids this routine produces, "the literal with the *smaller* component id is true." For variable $i$ with literals at vertices $2i$ and $2i+1$: it's true exactly when $\mathrm{comp}[2i] < \mathrm{comp}[2i+1]$. (If I'd used the kind of routine whose second pass numbers components in forward topological order, the comparison would flip to $>$; the principle is identical, only the id convention differs.) And the unsatisfiability test is just $\mathrm{comp}[2i] = \mathrm{comp}[2i+1]$ for some $i$ — same component.

One more practical hazard at this scale. The natural component routine is recursive depth-first search, but with up to $2 \times 10^6$ vertices a recursive DFS will blow the call stack — a long implication chain is a path that deep. So I'll run the depth-first search *iteratively*, with an explicit stack of (vertex, position-in-its-adjacency-list) frames, simulating the recursion: when I'd recurse into an unvisited neighbor I push a frame; when a vertex's neighbors are exhausted I do the low-link update against its parent and pop. The component-closing logic — when a vertex's low-link equals its own discovery index, pop the component off the auxiliary stack — is unchanged. Everything stays $O(n + m)$.

Let me also fix the literal encoding so the negation operation is trivial. Put variable $i$'s "true" literal at vertex $2i$ and its "false" literal at $2i+1$; then flipping a literal is just exclusive-or with $1$, $\mathrm{neg}(u) = u \oplus 1$, and a variable's two vertices are $2i$ and $2i+1$, adjacent integers. Building the implication graph is then: for each clause given as a pair of literal-vertices $(a, b)$, add edge $\mathrm{neg}(a) \to b$ and edge $\mathrm{neg}(b) \to a$. Reading input, a signed token $i$ or $-i$ (1-based, sign for negation) maps to $2(|i|-1) + [\,i<0\,]$.

So the whole pipeline: parse clauses into literal-vertex pairs; build the $2n$-vertex implication graph by adding the two implication edges per clause; run iterative strongly-connected-components to get a component id per vertex (ids in reverse topological order); for each variable, if its two literals share a component report unsatisfiable, otherwise set the variable true when its positive literal's component id is the smaller (the later, sink-ward one). Let me write it.

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

The chain, start to finish: a two-literal clause $(a\lor b)$ forbids only "both false," which is exactly the pair of implications $\lnot a \Rightarrow b$ and $\lnot b \Rightarrow a$, so the formula becomes a directed graph on $2n$ literal-vertices where reachability is forced truth and the edge set is closed under reverse-and-negate; a variable is hopeless precisely when its two literals are mutually reachable, i.e. share a strongly connected component, and that necessary condition is also sufficient because, when no variable's literals share a component, setting the later-in-topological-order literal of each variable to true can never force a literal to collide with its own negation — the reverse-and-negate symmetry would otherwise make $x$ reach $\lnot x$ across the topological order, which is impossible; computing components in one iterative pass (ids in reverse topological order) lets me test same-component for unsatisfiability and compare ids to read off the witness, all in $O(n + m)$.
