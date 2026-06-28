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

Single-file C++17. Reads from stdin: `n m`, then `m` clauses, each given as two signed literal tokens (variable `i` is 1-based; `+i` means `x_i` true, `-i` means `x_i` false). Prints `SATISFIABLE` and a line of `n` 0/1 values, or `UNSATISFIABLE`.

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

## Complexity

- **Time:** $O(n + m)$ — building the $2n$-vertex, $2m$-edge implication graph is $O(n + m)$, one SCC pass is $O(V + E) = O(n + m)$, the assignment scan is $O(n)$.
- **Memory:** $O(n + m)$ for the adjacency lists and the per-vertex arrays.
- The SCC pass is iterative so that an implication chain of length $\sim 2 \times 10^6$ does not overflow the recursion stack.
