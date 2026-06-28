# Counting ordered triples by their sum

## Research question

You are given a multiset of `n` non-negative integers `a[0..n-1]`, each in the range `[0, V]`. For a
target value `s`, define `T(s)` as the number of **ordered triples** of indices `(i, j, k)` with
`i, j, k ∈ {0, 1, ..., n-1}` — chosen **independently**, so repetitions of an index are allowed —
such that `a[i] + a[j] + a[k] = s`. Then you are given `q` query values `s_1, ..., s_q`, each in
`[0, 3V]`, and you must output `T(s_j) mod 998244353` for every query.

Concretely: how many ways can you pick three array elements (order matters, the same position may be
reused in more than one slot) so their values sum to exactly `s`? Because indices are independent, an
element of value `v` appearing `f` times contributes a frequency, and the count of triples summing to
`s` is a sum over all value-decompositions `u + w + x = s` of `f[u]·f[w]·f[x]`. The arithmetic is done
modulo the prime `998244353`.

This is the "count combinations by sum" subproblem that appears inside counting-DP, generating-function,
and additive-combinatorics tasks. The one-variable, three-slot version with `V` and `q` both large is
exactly where the naive "double loop over value pairs" stops being fast enough.

## Input / output contract

- Input (stdin):
  - Line 1: two integers `n` and `V` (`0 <= n <= 2*10^5`, `0 <= V <= 2*10^5`).
  - Line 2: `n` integers `a[i]` (`0 <= a[i] <= V`), whitespace-separated. (If `n = 0` this line is empty.)
  - Line 3: one integer `q` (`0 <= q <= 2*10^5`).
  - Line 4: `q` integers `s_j` (`0 <= s_j <= 3*V`), whitespace-separated. (If `q = 0` this line is empty.)
  - All tokens may be separated by arbitrary whitespace / newlines; read tokenwise.
- Output (stdout): `q` lines, the `j`-th being `T(s_j) mod 998244353`.
- Time limit: 2 seconds. Memory: 256 MB.

Worked sample. Input:

```
3 1
0 1 1
4
0 1 2 3
```

Here `a = [0, 1, 1]`, so the value frequencies are `f[0] = 1`, `f[1] = 2`. The ordered triples:
sum `0` has `1` way `(0,0,0)`; sum `1` has `6`; sum `2` has `12`; sum `3` has `8`. Output:

```
1
6
12
8
```

(Check `s = 3`: every slot must be a `1`, and there are `2` array elements equal to `1`, so `2^3 = 8`.)

## Background

Let `f` be the **frequency polynomial** of the array: `f(x) = Σ_v f[v] · x^v`, where `f[v]` counts how
many array elements equal `v`. Then `f(x)^2` is the polynomial whose `x^s` coefficient is the number of
ordered *pairs* summing to `s`, and `f(x)^3` has `x^s` coefficient equal to the number of ordered
*triples* summing to `s`. So `T(s) = [x^s] f(x)^3`, and the whole problem reduces to computing the cube
of a polynomial of degree up to `V` (giving a polynomial of degree up to `3V`), then reading off the
requested coefficients.

Two families of approach are on the table before committing:

- **Schoolbook (naive) convolution.** Compute `f^2` by a double loop over value pairs `(u, w)`, then
  multiply by `f` again. Each multiplication of polynomials of size up to `V+1` costs `O(V^2)`. Simple
  and obviously correct; the open question is whether `O(V^2)` survives `V = 2*10^5`.
- **Fast convolution via a transform.** A polynomial product is a convolution, and convolution becomes
  pointwise multiplication under a suitable transform. The open question is which transform keeps the
  answer *exact* modulo `998244353` and runs in `O(D log D)` for `D ~ 3V`.

## Evaluation settings

Judged on hidden tests covering: `n = 0` (empty array — every count is `0`); `V = 0` (all values equal
`0`, so the only nonzero count is at `s = 0` and equals `n^3 mod p`); single-element arrays; arrays that
stack many copies of one value (frequencies far exceeding `1`, so the `mod` actually bites); queries at
the boundaries `s = 0` and `s = 3V`; and large instances with `n = V = q = 2*10^5` so an `O(V^2)`
solution times out while an `O(V log V)` one does not.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 998244353;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n; long long V;
    if (!(cin >> n >> V)) return 0;

    vector<long long> f(V + 1, 0);
    for (int i = 0; i < n; i++) {
        long long x; cin >> x;
        f[x] = (f[x] + 1) % MOD;
    }

    // TODO: form the cube of the frequency polynomial f modulo MOD,
    // TODO: so that coefficient [x^s] equals the number of ordered triples summing to s.

    int q; cin >> q;
    for (int j = 0; j < q; j++) {
        long long s; cin >> s;
        long long ans = 0;
        // TODO: ans = coefficient at x^s of f^3 (0 if out of range)
        cout << ans << "\n";
    }
    return 0;
}
```
