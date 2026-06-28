# Context

## Problem

Given n boolean variables x_1..x_n and m clauses, each of the form (a OR b) where a, b are literals (a variable or its negation), decide whether the formula is satisfiable, and if so output a satisfying assignment. The deliverable is a single self-contained C++17 program that reads from stdin and writes to stdout. The input begins with `n m`, followed by `m` clauses, each given as two signed 1-based literal tokens (`i` or `+i` for x_i true, `-i` for x_i false). Output `UNSATISFIABLE`, or output `SATISFIABLE` followed by a line of `n` 0/1 assignment values. (2-SAT; n, m up to ~10^6.)

## Code framework

```cpp
#include <bits/stdc++.h>
using namespace std;

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

    // TODO
    bool satisfiable = false;
    vector<char> assign(n, 0);

    if (!satisfiable) {
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
