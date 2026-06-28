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

The deliverable is a single self-contained C++17 program reading the instance from stdin: the first line is `n m`, then `m` clauses each given as two signed literal tokens (variable `i` 1-based, `+i` for `x_i` true and `-i` for `x_i` false); it prints `SATISFIABLE` followed by a line of `n` 0/1 values, or `UNSATISFIABLE`. Vertex ids fit in `int` since $2n \le 2 \times 10^6$, but the signed input tokens are read as `long long` before mapping to a vertex so the encoding never overflows.

```cpp
// 2-SAT via strongly connected components.
// Reads from stdin: "n m", then m clauses, each two signed literal tokens
// (variable i is 1-based; +i means x_i true, -i means x_i false).
// Prints "SATISFIABLE" and a line of n 0/1 values, or "UNSATISFIABLE".
#include <cstdio>
#include <vector>
#include <string>
using namespace std;

// Iterative SCC. Fills comp[0..tot-1], the component id of each vertex.
// Two vertices share an id exactly when each is reachable from the other.
// Component ids follow the order in which components close, so a sink
// component (nothing downstream) closes first and gets the smaller id:
// the ids run in reverse topological order. O(V + E).
static void strongly_connected_components(int tot, const vector<vector<int>>& adj,
                                          vector<int>& comp) {
    vector<int> index(tot, 0), low(tot, 0), stk;
    vector<char> on_stack(tot, 0);
    comp.assign(tot, -1);
    stk.reserve(tot);
    // Explicit work stack of (vertex, adjacency position) to avoid recursion,
    // which would overflow at ~2e6 vertices on a long implication chain.
    vector<int> wv, wp;
    wv.reserve(tot);
    wp.reserve(tot);
    int counter = 0, cid = 0;
    for (int s = 0; s < tot; ++s) {
        if (index[s]) continue;
        wv.push_back(s);
        wp.push_back(0);
        while (!wv.empty()) {
            int v = wv.back();
            int& pi = wp.back();
            if (pi == 0) {
                index[v] = low[v] = ++counter;
                stk.push_back(v);
                on_stack[v] = 1;
            }
            bool recursed = false;
            while (pi < (int)adj[v].size()) {
                int u = adj[v][pi++];
                if (index[u] == 0) {
                    wv.push_back(u);
                    wp.push_back(0);
                    recursed = true;
                    break;
                } else if (on_stack[u] && index[u] < low[v]) {
                    low[v] = index[u];
                }
            }
            if (recursed) continue;
            if (low[v] == index[v]) {
                while (true) {
                    int w = stk.back();
                    stk.pop_back();
                    on_stack[w] = 0;
                    comp[w] = cid;
                    if (w == v) break;
                }
                ++cid;
            }
            int lv = low[v];
            wv.pop_back();
            wp.pop_back();
            if (!wv.empty()) {
                int pv = wv.back();
                if (lv < low[pv]) low[pv] = lv;
            }
        }
    }
}

// Solve n-variable 2-SAT. clauses holds 2m literal-vertices (a, b per clause),
// each literal encoded as a node: variable i positive is 2*i, negated 2*i+1.
// Returns true if satisfiable, filling assign[0..n-1]; false otherwise.
static bool solve(int n, const vector<int>& clauses, vector<char>& assign) {
    int tot = 2 * n;
    vector<vector<int>> adj(tot);
    for (size_t k = 0; k + 1 < clauses.size(); k += 2) {
        int a = clauses[k], b = clauses[k + 1];
        // (a OR b) is the two forced cases: not-a -> b and not-b -> a.
        adj[a ^ 1].push_back(b);
        adj[b ^ 1].push_back(a);
    }
    vector<int> comp;
    strongly_connected_components(tot, adj, comp);
    assign.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        int positive = 2 * i, negative = positive + 1;
        if (comp[positive] == comp[negative]) return false;
        // This SCC routine closes sinkward components first, so smaller id is later.
        assign[i] = (comp[positive] < comp[negative]) ? 1 : 0;
    }
    return true;
}

// Signed token i (1-based, sign for negation) -> literal-vertex.
static inline int lit_to_node(long long v) {
    long long var = (v < 0 ? -v : v) - 1;
    return (int)(2 * var + (v > 0 ? 0 : 1));
}

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<int> clauses;
    clauses.reserve((size_t)2 * m);
    for (int i = 0; i < m; ++i) {
        long long a, b;
        if (scanf("%lld %lld", &a, &b) != 2) break;
        clauses.push_back(lit_to_node(a));
        clauses.push_back(lit_to_node(b));
    }
    vector<char> assign;
    if (!solve(n, clauses, assign)) {
        printf("UNSATISFIABLE\n");
    } else {
        printf("SATISFIABLE\n");
        string out;
        out.reserve((size_t)2 * n);
        for (int i = 0; i < n; ++i) {
            if (i) out.push_back(' ');
            out.push_back(assign[i] ? '1' : '0');
        }
        out.push_back('\n');
        fputs(out.c_str(), stdout);
    }
    return 0;
}
```
