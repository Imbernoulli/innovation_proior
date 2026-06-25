**Reading the problem and pinning the contract.** I have `n` crystal shards, each with a mass `w[i]`
and a phase `p[i]`. I weld a subset `S` (the empty subset is allowed) and it resonates at `(W, q)`
exactly when `sum_{i in S} w[i] == W` *and* `(sum_{i in S} p[i]) mod M == q`. I must count the resonant
subsets modulo `1_000_000_007`. Distinctness is by *which shards*, not by their values, so two equal-mass
equal-phase shards still create two different subsets when one swaps for the other. Input is a header
`n W M q` then `n` lines of `w[i] p[i]`; output is one integer. Let me fix the scale before choosing an
algorithm: `n <= 200`, `W <= 2000`, `M <= 200`, `w[i] <= 2000`, and crucially `p[i] <= 10^9` — far
bigger than `M`. So the phases must be reduced mod `M` before anything else; a raw `p[i]` can be a
billion and I only ever care about `p[i] mod M`. Counts are taken mod `1e9+7`, so I work in `long long`
and reduce after every addition; the count itself can be astronomically large (up to `2^200` subsets),
so I never hold it un-reduced.

**Laying out the candidate approaches.** Two routes, and I want the one I can *prove*, not the one that
is shortest to type.

- *Factor the phase out with a closed form.* The seductive idea: first count `C = #{subsets with mass
  exactly W}` — a clean one-dimensional exact-subset-sum count — and then argue the phase residues of
  those `C` subsets are distributed in some assertable way, e.g. uniformly across the `M` residues, so
  the answer is `C / M`. If that held, the phase dimension would evaporate and I would do an `O(n*W)`
  count instead of an `O(n*W*M)` table. This is exactly the kind of "by symmetry it's uniform" step
  that *feels* obviously true and is the trap the problem is built around. I will not assert it; I will
  test it.
- *Carry both constraints in one DP table.* Track `dp[mass][residue] = #subsets with total mass exactly
  `mass` and total phase ≡ `residue (mod M)`. Process shards one at a time as a 0/1 knapsack. The answer
  is `dp[W][q]`. The risks here are mechanical: the 0/1 iteration order (forward vs backward over mass),
  and reducing `p[i]` mod `M` so the residue wrap is a single subtraction.

**Stress-testing the closed form before committing — and killing it numerically.** Let me actually
check the "residues are uniform" claim on a tiny concrete instance instead of trusting the symmetry
hunch. Take `n = 3`, `W = 2`, `M = 3`, shards `(w, p) = (1, 0), (1, 0), (2, 1)`. Which subsets reach
mass exactly `2`? Subset `{0, 1}` has mass `1 + 1 = 2`, phase `0 + 0 = 0`, residue `0`. Subset `{2}` has
mass `2`, phase `1`, residue `1`. No other subset reaches mass `2` (`{}` is mass 0, singletons `{0}`,
`{1}` are mass 1, `{0,2}` and `{1,2}` are mass 3, `{0,1,2}` is mass 4). So `C = 2` subsets hit mass `W`,
and their residues are: residue `0` -> 1 subset, residue `1` -> 1 subset, residue `2` -> **0** subsets.

The uniform closed form predicts each residue gets `C / M = 2 / 3 = 0.667` subsets. That is not even an
integer, and it disagrees with every actual cell: residues `0` and `1` have one each, residue `2` has
none. The distribution across residues is genuinely non-uniform and there is no clean divide-by-`M`. The
closed form is false, and I caught it before writing a line of the wrong solution. This is the whole
point: a tempting "it's uniform" assertion, disproved by enumerating one small case. I throw the
closed-form route out and commit to the coupled table.

(Why is uniformity false? The residue of a subset is `sum p[i] mod M`, but only over subsets that *also*
hit mass `W`. The mass constraint correlates which shards — hence which phases — can co-occur, so the
residues of the mass-`W` subsets carry no reason to be balanced. Uniformity would need a symmetry that
shifts residues while preserving mass, and no such symmetry exists in general. The numbers above are the
proof; the intuition is just commentary.)

**Deriving the coupled DP and checking the recurrence on paper.** I want
`dp[m][r] = #{subsets of the shards processed so far with total mass exactly `m` and total phase ≡ `r``.
Masses above `W` are useless — once a partial weld exceeds `W` it can never come back down — so I cap
the mass axis at `W`. The table is `(W + 1)` masses times `M` residues. Base case before any shard: only
the empty subset exists, with mass `0` and phase `0`, so `dp[0][0] = 1` and every other cell `0`.

Transition for shard `i` with mass `wi = w[i]` and reduced phase `pi = p[i] mod M`: a subset either
omits shard `i` (its `dp` value is unchanged) or includes it. Including it maps a previous subset of
mass `m - wi` and residue `r` to a new subset of mass `m` and residue `(r + pi) mod M`. So in 0/1 form
the update is, for every `m` from `W` down to `wi` and every residue `r`:

    dp[m][(r + pi) mod M] += dp[m - wi][r].

Iterating mass **downward** is what makes this 0/1 (each shard used at most once): when I update `dp[m]`
from `dp[m - wi]`, the source row `dp[m - wi]` has not yet been touched by shard `i` in this pass,
because I have not reached it yet (I am descending). If I iterated upward I would read a `dp[m - wi]`
that *already* absorbed shard `i`, which would let me weld the same shard twice — an unbounded knapsack,
which is wrong here.

Let me confirm the recurrence by hand on the sample: `n = 4`, `W = 5`, `M = 3`, `q = 2`, shards
`(2,1), (3,2), (2,0), (1,1)` (already reduced mod 3). I will track only the cells that ever become
nonzero. Start: `dp[0][0] = 1`. Shard 0 `(wi=2, pi=1)`: for `m = 2`, `dp[2][(0+1)%3] = dp[2][1] +=
dp[0][0] = 1`. Now nonzero cells: `dp[0][0]=1`, `dp[2][1]=1`. Shard 1 `(wi=3, pi=2)`: descend `m` from 5.
`m=5`: `dp[5][(1+2)%3]=dp[5][0] += dp[2][1] = 1`. `m=3`: `dp[3][(0+2)%3]=dp[3][2] += dp[0][0] = 1`. Now:
`dp[0][0]=1`, `dp[2][1]=1`, `dp[3][2]=1`, `dp[5][0]=1`. Shard 2 `(wi=2, pi=0)`: descend. `m=5`:
`dp[5][(2+0)%3]=dp[5][2] += dp[3][2] = 1`. `m=4`: `dp[4][(1+0)%3]=dp[4][1] += dp[2][1] = 1`. `m=2`:
`dp[2][(0+0)%3]=dp[2][0] += dp[0][0] = 1`. Now nonzero: `dp[0][0]=1, dp[2][1]=1, dp[2][0]=1, dp[3][2]=1,
dp[4][1]=1, dp[5][0]=1, dp[5][2]=1`. Shard 3 `(wi=1, pi=1)`: descend. `m=5`: `dp[5][(1+1)%3]=dp[5][2] +=
dp[4][1] = 1+1 = 2`. `m=4`: `dp[4][(2+1)%3]=dp[4][0] += dp[3][2] = 1`. `m=3`: `dp[3][(1+1)%3]=dp[3][2] +=
dp[2][1] = 1+1 = 2`; also `dp[3][(0+1)%3]=dp[3][1] += dp[2][0] = 1`. `m=1`: `dp[1][(0+1)%3]=dp[1][1] +=
dp[0][0] = 1`. The answer cell is `dp[W][q] = dp[5][2]`, which is now `2`. That matches the expected
sample answer of `2`, and the two contributions to `dp[5][2]` are exactly the two subsets named in the
statement. The recurrence is right.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut
of the core loop, trying to be tidy:

```
for (int i = 0; i < n; i++) {
    int wic = (int)w[i], pi = p[i] % M;
    if (wic > Wcap) continue;
    for (int m = wic; m <= Wcap; m++) {          // <-- forward
        for (int ph = 0; ph < M; ph++) {
            long long v = dp[m - wic][ph];
            if (!v) continue;
            int nph = ph + pi; if (nph >= M) nph -= M;
            dp[m][nph] = (dp[m][nph] + v) % MOD;
        }
    }
}
```

Something about iterating mass *upward* and reading `dp[m - wic]` while writing `dp[m]` nags at me, so I
trace the smallest input that exposes 0/1-vs-unbounded confusion: two identical shards `(mass 1, phase
0)`, `W = 2`, `M = 1`, `q = 0`. The only resonant subset is `{both}` (mass `2`), so the answer must be
`1`. Trace: `dp[0][0] = 1` to start, `M = 1` so the only residue is `0`. Shard 0 `(wi=1, pi=0)`, forward
`m` from 1: `m=1`: `dp[1][0] += dp[0][0] = 1`. `m=2`: `dp[2][0] += dp[1][0] = 1`. Now `dp[0][0]=1,
dp[1][0]=1, dp[2][0]=1`. Shard 1 `(wi=1, pi=0)`, forward `m` from 1: `m=1`: `dp[1][0] += dp[0][0] =
1+1=2`. `m=2`: `dp[2][0] += dp[1][0] = 1 + 2 = 3`. Final `dp[2][0] = 3`.

**Diagnosing the first bug.** The code returns `3`, but the true answer is `1`. The defect is precise:
on shard 1, iterating `m` upward, I first bumped `dp[1][0]` to `2` (that `2` already includes "shard 1
welded onto mass 0"), and then at `m=2` I read that *just-updated* `dp[1][0]=2` and added it into
`dp[2][0]`. So `dp[2][0]` absorbed a path that welds shard 1 onto a mass-1 subset that *also already
contains shard 1* — the same physical shard used twice. Forward iteration turned my intended 0/1
knapsack into an unbounded one. The fix is to descend `m` from `Wcap` down to `wic`, so the source row
`dp[m - wic]` is always one this pass has not yet modified.

**Fixing the iteration order and re-verifying.** Reverse the mass loop:

```
for (int m = Wcap; m >= wic; m--) {
    for (int ph = 0; ph < M; ph++) {
        long long v = dp[m - wic][ph];
        if (!v) continue;
        int nph = ph + pi; if (nph >= M) nph -= M;
        dp[m][nph] = (dp[m][nph] + v) % MOD;
    }
}
```

Re-trace the same `[(1,0),(1,0)]`, `W=2`, `M=1`. `dp[0][0]=1`. Shard 0, descend from 2: `m=2`: `dp[2][0]
+= dp[1][0] = 0`; `m=1`: `dp[1][0] += dp[0][0] = 1`. Now `dp[0][0]=1, dp[1][0]=1`. Shard 1, descend from
2: `m=2`: `dp[2][0] += dp[1][0] = 1`; `m=1`: `dp[1][0] += dp[0][0] = 1+1=2`. Final `dp[2][0] = 1`.
Correct — exactly the single subset `{both}`. The case that broke now passes, and it broke for the reason
I fixed (upward iteration double-welding a shard), which is the evidence I trust.

**Second bug — a trace that exposes the phase reduction.** With the order fixed I look hard at the
residue arithmetic, because `p[i]` can be up to `10^9` while `M <= 200`. My loop wrapped with a *single*
conditional subtraction: `int nph = ph + pi; if (nph >= M) nph -= M;`. That single subtraction is only
valid when `pi < M`, because then `ph + pi < 2M` and one subtraction suffices. If I forget to reduce
`pi = p[i] % M` and feed the raw phase, `pi` can be `10^9`; then `nph = ph + pi` is gigantic and one
subtraction leaves it `>= M`, so `dp[m][nph]` indexes far outside the residue array. Let me trace a case
that triggers it: one shard `(w=2, p=5)`, `W=2`, `M=3`, `q=0`. With the *raw* phase bug (`pi = 5`, no
`% M`): shard 0, `m=2`: `nph = 0 + 5 = 5`, single subtraction gives `5 - 3 = 2`... in this row `ph=0`
only, so `nph=2`, which happens to be in range `[0,3)` here, but it lands in the *wrong* residue: the
true residue is `5 mod 3 = 2`, which by luck equals `2`, so this one is not yet visibly wrong. Push
harder: two shards each `(w=2, p=5)`, `W=4`, `M=3`, `q=1`. True: subset `{both}` has mass `4`, phase
`5 + 5 = 10`, residue `10 mod 3 = 1 == q`, so the answer is `1`. With the raw-phase bug, after shard 0
the nonzero source has `ph=2` (because `5-3=2`); shard 1 at `m=4` computes `nph = 2 + 5 = 7`, a single
subtraction gives `7 - 3 = 4`, and `dp[4][4]` is an out-of-bounds write into an `M=3` row. Under address
sanitizing this is a hard heap-buffer-overflow; without it, silent corruption.

**Diagnosing and fixing the second bug.** Two coupled facts make the single subtraction correct: I must
reduce `pi = p[i] % M` *before* the loop, so `0 <= pi < M`, and then `0 <= ph < M` guarantees `ph + pi <
2M`, so exactly one conditional subtraction normalizes the residue back into `[0, M)`. I add the
reduction (`int pi = p[i] % M;`) and keep the single-subtraction wrap. Re-trace the two-shard case
`(2,5),(2,5)`, `W=4`, `M=3`, `q=1`, now with `pi = 5 % 3 = 2`. `dp[0][0]=1`. Shard 0 descend: `m=2`:
`nph = 0 + 2 = 2`, no subtract, `dp[2][2] += dp[0][0] = 1`. Shard 1 descend: `m=4`: `nph = 2 + 2 = 4 >=
3 -> 1`, `dp[4][1] += dp[2][2] = 1`; `m=2`: `nph = 0 + 2 = 2`, `dp[2][2] += dp[0][0] = 1+1 = 2`. Answer
`dp[4][1] = 1`. Correct, no out-of-bounds, residue `10 mod 3 = 1` recovered exactly. The phase axis is
now sound for phases up to `10^9`.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the shard loop never runs; the table stays at its base `dp[0][0] = 1`. The answer is
  `dp[W][q]`. If `W = 0` and `q = 0` this is `1` (the empty subset resonates); for any other `(W, q)` it
  is `dp[W][q] = 0`. I trace `n=0, W=0, M=5, q=0 -> 1` and `n=0, W=0, M=5, q=3 -> 0`. Correct: the bare
  mount has mass 0 and phase 0, residue 0, so it resonates only at `q = 0`.
- `M = 1`: the only residue is `0`, and `q` must be `0` by the contract `0 <= q < M`. Every subset's
  phase reduces to residue `0`, so the count collapses to "subsets with mass exactly `W`", regardless of
  phases. I trace `n=3, W=3, M=1, q=0` with phases `7, 0, 99` (each `% 1 = 0`): the answer is the number
  of three-shard subsets of unit masses summing to 3, which is `1` (all three). Confirmed `1`.
- A shard with `w[i] > W`: the `if (wic > Wcap) continue;` skips its inner update, which is correct
  precisely because skipping the update means the shard is simply never welded in — it still contributes
  its "not taken" branch (the existing `dp` cells), which is exactly its only feasible role. Note the
  cap also lets me store the table as `int`-indexed since `W <= 2000`.
- `q` with answer `0`: e.g. the sample masses but `q = 1` — the cell `dp[5][1]` stays `0`. The code
  prints `0`, no special-casing needed.
- Modulus and overflow: every cell stays in `[0, MOD)` because I reduce after each addition; the count
  can be enormous (up to `2^200` subsets) but I only ever hold it mod `1e9+7`. `dp` is `long long`, and
  the largest intermediate before reduction is `< 2 * MOD < 2^31`, far inside `long long`.
- Time and memory: the table is `(W+1) * M <= 2001 * 200 ~ 4 * 10^5` longs (~3.2 MB), and the work is
  `O(n * W * M) <= 200 * 2000 * 200 = 8 * 10^7` inner steps, which runs in well under a second (measured
  ~0.04 s on the full-size case). The `if (v == 0) continue;` skips empty source cells, which only helps.

**Final solution.** I disproved the tempting "residues are uniform, answer = C/M" closed form by
enumerating one small case (residues `0,1,2` got counts `1,1,0`, not `2/3` each), so I committed to the
coupled `dp[mass][residue]` table; I checked its recurrence by hand on the sample reaching `dp[5][2]=2`;
a trace of `[(1,0),(1,0)]` returning the illegal `3` exposed forward mass-iteration double-welding a
shard, fixed by descending; and a trace of `[(2,5),(2,5)]` exposed the missing `p[i] % M` reduction
blowing past the residue array, fixed by reducing the phase before the single-subtraction wrap. That is
what I ship — one self-contained file, the `O(n*W*M)` two-dimensional counting knapsack I can defend:

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long W;
    int M, q;
    if (!(cin >> n >> W >> M >> q)) return 0;

    vector<long long> w(n);
    vector<int> p(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> p[i];

    // Masses beyond W are useless; cap the table at W.
    // dp[mass][phase] = number of subsets with total mass == mass and total
    // phase (mod M) == phase. Two dimensions are coupled, so we carry both.
    int Wcap = (int)W; // W fits in int by constraints (<= 2000)
    vector<vector<long long>> dp(Wcap + 1, vector<long long>(M, 0));
    dp[0][0] = 1; // empty subset

    for (int i = 0; i < n; i++) {
        long long wi = w[i];
        int pi = p[i] % M;
        if (wi > Wcap) continue; // cannot fit, skip (still counts as "not taken")
        int wic = (int)wi;
        // 0/1 knapsack: iterate mass downward to avoid reusing item i.
        for (int m = Wcap; m >= wic; m--) {
            const vector<long long> &src = dp[m - wic];
            vector<long long> &dst = dp[m];
            for (int ph = 0; ph < M; ph++) {
                long long v = src[ph];
                if (v == 0) continue;
                int nph = ph + pi;
                if (nph >= M) nph -= M;
                dst[nph] += v;
                if (dst[nph] >= MOD) dst[nph] -= MOD;
            }
        }
    }

    cout << dp[Wcap][q] % MOD << "\n";
    return 0;
}
```

**Causal recap.** The phase dimension *looked* separable, but enumerating `n=3, W=2, M=3` with shards
`(1,0),(1,0),(2,1)` showed the mass-`W` subsets land in residues `0,1,2` with counts `1,1,0` — flatly
not the `C/M = 2/3` a uniform closed form predicts, so I refused the closed form and built the coupled
`dp[mass][residue]` table; its two transition branches (omit / include shard `i`) both read the previous
table, and my first upward mass loop read a row already touched in the same pass, double-welding a shard
(a trace of two unit shards returning the illegal `3` pinpointed it), fixed by descending the mass axis;
and the residue wrap is only a single subtraction once `p[i]` is reduced mod `M`, which a trace of two
phase-`5` shards (raw phase indexing past an `M=3` row) forced me to add; with `dp[W][q]` read at the
end, the empty-subset (`n=0`), vacuous-phase (`M=1`), oversized-mass, and modular-overflow corners all
fall out of the recurrence without special cases.
