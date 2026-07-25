# Superset counting over the subset lattice

## Research question

You are given `n` items, each described by a `B`-bit bitmask (an integer in `[0, 2^B)`), and `q`
query masks. For each query mask `m` you must report how many of the `n` items are **supersets** of
`m` — that is, how many items `x` satisfy `x & m == m` (every bit set in `m` is also set in `x`).

The mathematical object underneath is the **subset lattice** on `B` bits ordered by inclusion. The
multiset of items induces a counting function `cnt[t]` = number of items equal to mask `t`; the query
asks for the **up-set sum** of `cnt` at `m`, i.e. the sum of `cnt[t]` over all `t` with `t ⊇ m`.

## Input / output contract

- Input (stdin):
  - Line 1: three integers `B n q`.
  - Line 2: `n` integers, the item masks `x_i` (each in `[0, 2^B)`). May be empty when `n = 0`.
  - Line 3: `q` integers, the query masks `m_j` (each in `[0, 2^B)`).
- Output (stdout): `q` lines; line `j` is the number of items that are supersets of `m_j`.
- Constraints: `0 <= B <= 20`, `0 <= n <= 10^6`, `1 <= q <= 10^6`.
- Time limit: 2 seconds. Memory: 256 MB.

Example:

```
3 6 4
7 6 3 5 4 1
3 1 0 7
```

The items are `111, 110, 011, 101, 100, 001`. Query `3 = 011` needs bits 0 and 1 set: only `111`
and `011` qualify → `2`. Query `1 = 001` needs bit 0: `111, 011, 101, 001` → `4`. Query `0` (empty
mask) is satisfied by everything → `6`. Query `7 = 111` needs all three bits: only `111` → `1`. So
the output is:

```
2
4
6
1
```

## Evaluation settings

Judged on hidden tests covering: `B = 0` (only the empty mask exists); `n = 0` (no items, every
answer `0`); queries equal to the empty mask `0` (answer is always `n`); queries equal to the full
mask `2^B - 1` (answer is the count of items equal to the full mask); heavy duplicate item masks;
and large instances `B = 20`, `n = q = 10^6` with masks drawn across the whole cube.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int B, n, q;
    if (!(cin >> B >> n >> q)) return 0;

    const int SZ = 1 << B;
    vector<long long> f(SZ, 0);
    for (int i = 0; i < n; i++) {
        int x; cin >> x;
        f[x] += 1;
    }

    // TODO: turn f[mask] into the number of items that are supersets of `mask`,
    //       then answer each of the q queries in O(1).

    for (int i = 0; i < q; i++) {
        int m; cin >> m;
        cout << /* answer for m */ 0 << "\n";
    }
    return 0;
}
```
