# Counting resonant frequency pairs by a target residue

## Research question

A test bench holds `n` oscillators; oscillator `i` runs at integer frequency `a[i]` (hertz). The
controller declares two distinct oscillators `i` and `j` a **resonant pair** when their combined
frequency lands on a fixed residue class modulo a tuning constant `m` — precisely, when
`(a[i] + a[j]) mod m == t` for a given target residue `t` with `0 <= t < m`. Count how many
**unordered** pairs `{i, j}` (with `i != j`) are resonant, and output that count.

This is the residue-class version of "count pairs whose sum satisfies a divisibility condition"
(the special case `t = 0` asks for sums divisible by `m`). It is the kind of counting subproblem that
appears inside hashing, collision analysis, and additive-combinatorics tasks, so getting the
bucket arithmetic — and the data types that hold the counts — exactly right matters.

## Input / output contract

- Input (stdin): the first line has three integers `n`, `m`, `t`
  (`0 <= n <= 2*10^5`, `1 <= m <= 10^6`, `0 <= t < m`).
  The second line has `n` integers `a[i]` (`0 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the number of unordered resonant pairs.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `m = 5`, `t = 3`, and frequencies `a = [7, 1, 4, 6, 9, 2]` the answer is `5`.

## Background

Testing the condition pair by pair is `O(n^2)`, which is `4*10^10` operations at the top of the
range — far too slow. Two observations drive the fast approach, and one decision must be settled
before any of it:

- **The condition depends only on residues.** Whether `{i, j}` is resonant depends solely on
  `a[i] mod m` and `a[j] mod m`. So if `cnt[r]` is the number of frequencies congruent to `r`
  modulo `m`, every pair is determined by its residue pair `(r, s)`. There are at most `m` residues.
- **Counts and the answer can be astronomically large.** A single residue bucket can hold up to
  `n = 2*10^5` frequencies, and the number of resonant pairs can approach `C(n, 2) ~ 2*10^10`. Both
  the per-bucket products and the running total exceed the `~2.1*10^9` range of a 32-bit integer, so
  the data type that holds them is part of the problem, not an afterthought.

## Evaluation settings

Judged on hidden tests covering: `m = 1` (every pair resonant when `t = 0`), `t > 0` targets,
buckets that pair with themselves (`2r == t (mod m)`), the empty bench (`n = 0`), a single oscillator
(`n = 1`, no pairs), frequencies up to `10^9`, and large `n = 2*10^5` with a single dominant residue
bucket so the answer is near `2*10^10` (forcing 64-bit accumulation).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long m, t;
    if (!(cin >> n >> m >> t)) return 0;
    vector<long long> cnt(m, 0);
    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        cnt[((x % m) + m) % m]++;
    }

    // TODO: count unordered pairs {i, j} with (a[i] + a[j]) % m == t,
    // using the residue buckets cnt[0..m-1].
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
