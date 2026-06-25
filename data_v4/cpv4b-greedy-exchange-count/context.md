# Fewest floodlight arcs to light a ring road

## Research question

A circular ring road is divided into `m` evenly spaced kilometre markers numbered `0, 1, ..., m-1`.
The markers are arranged on a circle, so marker `m-1` is immediately followed by marker `0` again
(all marker arithmetic is modulo `m`).

A maintenance crew owns `n` portable floodlights. Floodlight `i` is described by a start marker
`s_i` and an arc length `len_i`: switched on, it illuminates the `len_i` consecutive markers

```
s_i, s_i + 1, s_i + 2, ..., s_i + len_i - 1   (all taken mod m)
```

i.e. a contiguous arc that runs *clockwise* from `s_i`. Each `len_i` satisfies `1 <= len_i <= m`, so
a floodlight lights at least one marker and at most the entire ring.

You must switch on a subset of the floodlights so that **every** marker `0..m-1` is illuminated by
at least one switched-on floodlight. Output the **minimum number of floodlights** that achieves this,
or `-1` if the whole ring can never be lit no matter how many you switch on.

This is the **minimum circular-arc cover** problem. It is a greedy-exchange problem: the optimal
"extend to the floodlight that reaches furthest" rule is provable by an exchange argument, but the
circle (rather than a line) makes the *counting* delicate — a wrapping arc straddles the seam at
marker `0`, and it is dangerously easy to count one floodlight twice.

## Input / output contract

- Input (stdin): the first line has two integers `m` and `n`
  (`1 <= m <= 10^9`, `1 <= n <= 2000`).
  Each of the next `n` lines has two integers `s_i len_i`
  (`0 <= s_i <= m-1`, `1 <= len_i <= m`).
- Output (stdout): a single line with the minimum number of floodlights needed to light every
  marker, or `-1` if it is impossible.
- Time limit: 1 second. Memory: 256 MB.

Example: for

```
8 5
0 3
2 3
5 2
6 3
7 2
```

the answer is `4`. The five arcs light `{0,1,2}`, `{2,3,4}`, `{5,6}`, `{6,7,0}`, `{7,0}`
respectively. One optimal choice is `{0,1,2} + {2,3,4} + {5,6} + {6,7,0}`, which together cover all
of `0..7`; no three of the arcs suffice.

## Background

Two ingredients combine here.

- **Greedy interval cover on a line** is classical: to cover a segment with the fewest intervals,
  repeatedly jump from the current covered endpoint to the interval that starts no later than that
  endpoint and reaches furthest. An exchange argument shows this is optimal.
- **The circle** breaks the clean "leftmost point" anchor a line gives you, because there is no
  leftmost marker. The standard fix is to *pin down* one marker that must be covered (say marker
  `0`), enumerate which arc covers it *first*, and for each such choice unroll the ring into a line
  window of length `m` and run the line greedy. The minimum over those choices is the answer.

The trap is in the unrolling. A floodlight that wraps past the seam (`s_i + len_i > m`) contributes
coverage both at the high end of the window and near marker `0`; if it is represented twice in the
sweep without care, the greedy can *count the same physical floodlight as two switched-on lights*,
inflating the answer. Constraints are chosen so this double-count actually changes outputs.

## Evaluation settings

Judged on hidden tests covering: a single full-ring arc (`len = m`, answer `1`); genuinely
infeasible inputs (answer `-1`); `m = 1`; many arcs that all wrap across the seam; dense tilings
where the optimum is large; duplicate and fully-redundant arcs (which must not be double-counted);
and worst-case `n = 2000` with `m` near `10^9` so marker coordinates exceed 32 bits and every arc
is an anchor candidate.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long m;
    int n;
    if (!(cin >> m >> n)) return 0;

    vector<long long> S(n), Ln(n);
    for (int i = 0; i < n; i++) {
        long long s, L;
        cin >> s >> L;
        s %= m; if (s < 0) s += m;
        S[i] = s; Ln[i] = L;
    }

    // TODO: minimum number of arcs whose union covers all m markers, else -1.
    long long answer = -1;

    cout << answer << "\n";
    return 0;
}
```
