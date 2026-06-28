## Problem

You are given a directed graph on $d$ vertices, where each ordered pair $(i, j)$ carries a nonnegative integer count of edges from vertex $i$ to vertex $j$. Fix a source vertex $s$, a target vertex $t$, and an integer $N$. Count the number of **walks of length exactly $N$** from $s$ to $t$ — sequences $s = u_0 \to u_1 \to \dots \to u_N = t$ where each $u_{r-1} \to u_r$ is an edge, walks that revisit vertices or reuse edges are allowed and counted with multiplicity, and each step's choice is independent.

A second input form gives a $d$-term linear recurrence
$$a_n = c_1\, a_{n-1} + c_2\, a_{n-2} + \dots + c_d\, a_{n-d}$$
with fixed integer coefficients $c_1, \dots, c_d$ and $d$ given initial terms $a_0, a_1, \dots, a_{d-1}$ (a generalized Fibonacci sequence), and you must evaluate $a_N$.

The length or index $N$ can be as large as $10^{18}$. To keep the numbers bounded, every answer is reported modulo a given positive integer $m$ (the exact counts can grow extremely quickly, so the value modulo $m$ is what is required).

## Research question / Input-output contract

The deliverable is a single self-contained C++17 program that reads from stdin
and writes the answer to stdout. The first token selects the input form:
`W` then `d s t N m` followed by the $d \times d$ edge-count table prints the
length-$N$ walk count from `s` to `t`; `L` then `d N m`, followed by the $d$
coefficients and then the $d$ initial terms, prints $a_N$. The program prints
one integer, reduced modulo `m`.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string mode;
    if (!(cin >> mode)) return 0;

    long long answer = 0;
    if (mode == "W" || mode == "w") {
        int d, s, t;
        long long N, m;
        cin >> d >> s >> t >> N >> m;

        vector<vector<long long>> adj(d, vector<long long>(d));
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                cin >> adj[i][j];
            }
        }

        // TODO: compute the requested answer.
    } else {
        int d;
        long long N, m;
        cin >> d >> N >> m;

        vector<long long> coeffs(d), init(d);
        for (int i = 0; i < d; ++i) cin >> coeffs[i];
        for (int i = 0; i < d; ++i) cin >> init[i];

        // TODO: compute the requested answer.
    }

    cout << answer << '\n';
    return 0;
}
```
