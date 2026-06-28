# Counting distinct bracelets under rotation and reflection

## Research question

A bracelet has `n` beads arranged in a circle, and each bead is given one of `k` colors. Two
colorings are considered **the same bracelet** if one can be turned into the other by a rotation of
the circle and/or a flip (reflection) of the bracelet — i.e. by any element of the dihedral group
`D_n` acting on the `n` positions. Count the number of **distinct** bracelets and output it modulo
`p = 1000000007`.

The twist is scale: `n` can be as large as `10^9`, so the answer itself is astronomically large and
must be returned modulo a prime, and — more importantly — any method that walks over the `n` rotations
one at a time is hopeless. The interesting question is whether the per-symmetry contributions can be
*regrouped* so that the whole rotation part is computed from the divisor structure of `n` rather than
from `n` individual rotations.

## Input / output contract

- Input (stdin): two integers `n` and `k` on one line, `1 <= n <= 10^9` and `1 <= k <= 10^9`.
- Output (stdout): a single line with the number of distinct bracelets, taken modulo
  `p = 1000000007`.
- Time limit: 1 second. Memory: 256 MB.

Examples:

- `n = 4, k = 2` -> `6`. (The six distinct 2-color bracelets on 4 beads.)
- `n = 6, k = 3` -> `92`.
- `n = 1000000000, k = 2` -> `537849669` (the true count taken mod `p`).

## Background

The object that makes "same up to symmetry" precise is **Burnside's lemma** (the Cauchy–Frobenius
counting lemma): the number of orbits of a finite group `G` acting on a set equals the average, over
all `g in G`, of the number of configurations fixed by `g`. For bracelets the group is the dihedral
group `D_n` with `2n` elements — `n` rotations and `n` reflections.

Two ingredients are needed before committing to an implementation:

- **Counting fixed colorings of a rotation.** A rotation by `d` positions splits the `n` beads into
  `gcd(d, n)` cycles; a coloring is fixed iff every cycle is monochromatic, so it has `k^{gcd(d,n)}`
  fixed colorings. Summed naively over `d = 0..n-1` this is `O(n)` modular exponentiations — far too
  slow at `n = 10^9`. The open question is how to collapse this sum.

- **Counting fixed colorings of a reflection.** A reflection's cycle count depends on the parity of
  `n` and on whether the mirror axis passes through beads or through gaps. There are only `O(1)`
  distinct reflection types, so this part is cheap once the cases are pinned down correctly.

## Evaluation settings

Judged on hidden tests covering: tiny `n` and `k` (where an orbit-enumeration brute force is the
ground truth), both parities of `n`, `k = 1` (a single bracelet), prime `n`, highly composite `n`
(many divisors), `n` near `10^9`, and `k` near `10^9` (so `k` must be reduced modulo `p` before any
exponentiation). The answer is checked exactly against the modular count.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll MOD = 1000000007LL;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    ll n, k;
    if (!(cin >> n >> k)) return 0;

    // TODO: count distinct bracelets on n beads with k colors under the
    // dihedral group D_n (rotations + reflections), via Burnside's lemma,
    // and print the count modulo MOD.
    ll answer = 0;

    cout << answer << "\n";
    return 0;
}
```
