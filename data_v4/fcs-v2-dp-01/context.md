# N-th term of a long linear recurrence modulo a prime

## Research question

A sequence `a[0], a[1], a[2], ...` over the field `Z_p` (`p = 998244353`) satisfies an order-`k`
linear recurrence with fixed coefficients `c[0..k-1]`:

```
a[n] = c[0]*a[n-1] + c[1]*a[n-2] + ... + c[k-1]*a[n-k]   (mod p)   for n >= k.
```

The first `k` terms `a[0..k-1]` (the *seeds*) are given. Given a single index `N`, compute `a[N] mod p`.

The catch is the scale. The order `k` is large (up to `2*10^4`) and the index `N` is astronomically
large (up to `10^18`). We cannot generate the sequence term by term — that would be `Theta(N*k)`
operations, hopeless for `N = 10^18`. The question is how to jump directly to the `N`-th term using
only structure of the recurrence.

## Input / output contract

- Input (stdin), whitespace-separated in this order:
  1. an integer `k` (`1 <= k <= 2*10^4`), the order of the recurrence;
  2. `k` integers `c[0], c[1], ..., c[k-1]` — the recurrence coefficients, each given modulo `p`
     (the input value may be any integer; reduce it mod `p = 998244353`, taking the nonnegative
     representative);
  3. `k` integers `a[0], a[1], ..., a[k-1]` — the seed terms, likewise read modulo `p`;
  4. an integer `N` (`0 <= N <= 10^18`), the index of the term to compute.
- Output (stdout): a single line with `a[N] mod p`, as an integer in `[0, p)`.
- Time limit: 3 seconds. Memory: 256 MB.

Worked example. The Fibonacci recurrence has `k = 2`, `c = [1, 1]` (so `a[n] = a[n-1] + a[n-2]`)
and seeds `a = [0, 1]`. For `N = 10` the answer is `55`. For `N = 0` it is `0`; for `N = 1` it is `1`.

## Background

For tiny `k` the textbook tool is **matrix exponentiation**: stack the last `k` terms into a state
vector and advance one step by multiplying with the `k x k` companion matrix of the recurrence;
`a[N]` then falls out of `M^N` applied to the seed state, computed by fast exponentiation in
`O(k^3 log N)` field operations. That cubic factor in `k` is fine for `k <= 100` or so, but with
`k = 2*10^4` it is `(2*10^4)^3 = 8*10^{13}` per matrix multiply — completely out of reach.

There is a second, equivalent way to phrase "advance the recurrence" that does not build a matrix at
all. Associate with the recurrence its **characteristic polynomial**

```
g(x) = x^k - c[0]*x^{k-1} - c[1]*x^{k-2} - ... - c[k-1].
```

The state-advance map is multiplication by `x` in the quotient ring `Z_p[x] / (g(x))`. So computing
the `N`-step advance is computing `x^N` inside that ring — i.e. the polynomial `f(x) = x^N mod g(x)`,
which has degree `< k`. Once we have `f(x) = sum_{i=0}^{k-1} f[i]*x^i`, linearity gives

```
a[N] = sum_{i=0}^{k-1} f[i] * a[i]   (mod p),
```

a single dot product of the reduced polynomial with the seed vector. This reduces the whole task to
"raise `x` to the `N`-th power modulo a degree-`k` polynomial," which is the object the rest of the
work has to make fast.

## Evaluation settings

Judged on hidden tests covering: `k = 1` (geometric sequences); `N < k` (the answer is a raw seed);
`N` exactly at the boundaries `k-1`, `k`; recurrences with zero / sparse coefficients (so the
characteristic polynomial has many zero terms); coefficients and seeds spanning the full range
`[0, p)`; and the stress corner `k = 2*10^4`, `N = 10^18` with random full-range data, where only an
`O(k log k log N)` method finishes inside the time limit. All arithmetic is modulo
`p = 998244353`, an NTT-friendly prime.

## Code framework

A single self-contained C++17 program reading stdin and writing stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int k;
    if (!(cin >> k)) return 0;
    vector<long long> c(k), a(k);
    for (auto &x : c) cin >> x;   // recurrence coefficients (reduce mod p)
    for (auto &x : a) cin >> x;   // seed terms (reduce mod p)
    long long N;
    cin >> N;

    // TODO: compute a[N] mod p for the order-k recurrence with these coefficients and seeds.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
