# Context

## Problem

You are given the first $2k$ terms of a sequence over the field $\mathbb{Z}_p$ ($p$
prime). It is known to satisfy some linear recurrence of order $\le k$. Recover the
shortest such recurrence, then compute the $N$-th term ($N$ up to $10^{18}$) modulo
$p$.

A linear recurrence of order $m$ is a coefficient list $c_0, \dots, c_{m-1}$ such
that every term is the same fixed combination of the $m$ before it:
$$a_i = \sum_{j=0}^{m-1} c_j \, a_{i-j-1} \qquad \text{for all } i \ge m.$$
"Shortest" means smallest order $m$. The order is unknown in advance; only the
bound $m \le k$ is given, and the $2k$ supplied terms are exactly enough to pin a
recurrence of order $\le k$ down uniquely. The index $N$ can be as large as
$10^{18}$.

All arithmetic is modulo the prime $p$; division is multiplication by a modular
inverse.

## Code framework

The deliverable is a single self-contained C++17 program that reads from stdin
and writes to stdout. The input is read as whitespace-separated integers: $p$,
then $N$, then the count of supplied terms, then the terms themselves. The
program must print the $N$-th term modulo $p$ as a single integer.

```cpp
#include <bits/stdc++.h>
using namespace std;
int main(){
    // read input from stdin per the contract
    long long p, N;
    int cnt;
    if (!(cin >> p >> N >> cnt)) return 0;
    vector<long long> terms(cnt);
    for (int i = 0; i < cnt; ++i) cin >> terms[i];

    // TODO: implement
    long long answer = 0;

    // print to stdout
    cout << answer << '\n';
    return 0;
}
```
