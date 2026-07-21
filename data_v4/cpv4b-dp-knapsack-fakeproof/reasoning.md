Two constraints have to hold *at the same time*: a welded subset resonates only when its total mass
equals `W` exactly and its total phase, reduced mod `M`, equals `q`. That "at the same time" is the
whole difficulty — each constraint alone is a routine count, and the problem is clearly built to tempt
me into treating them as if they factor. I pin the scale first, because it decides everything: `n <=
200`, `W <= 2000`, `M <= 200`, `w[i] <= 2000`, and `p[i] <= 10^9`. That last one is the sharp corner —
a phase can be a billion while `M` is at most 200, so `p[i]` must be reduced mod `M` before it ever
touches an array index; only `p[i] mod M` matters. Counts run mod `1e9+7`, the raw count can be as
large as `2^200`, so I hold everything reduced in `long long` and never carry an un-reduced total.
Input is a header `n W M q` then `n` lines `w[i] p[i]`; output is one integer. Distinctness is by
*which shards*, not by value, so two equal-mass equal-phase shards genuinely make two subsets.

The tempting shortcut, and the reason this problem exists, is to factor the phase out: count `C =
#{subsets with mass exactly W}` — a clean one-dimensional exact-subset-sum count — then claim those `C`
subsets spread uniformly across the `M` residues, so the answer is `C / M`. If that held I could do an
`O(n*W)` count and skip the phase axis entirely. It has the flavor of an obvious symmetry, which is
usually where these traps live. Take `n = 3`, `W = 2`, `M = 3`, shards
`(1,0), (1,0), (2,1)`. The subsets that reach mass exactly `2` are `{0,1}` (phase `0`, residue `0`) and
`{2}` (phase `1`, residue `1`); nothing else hits mass `2`. So `C = 2`, and the residue counts are
`(1, 1, 0)`. The uniform form predicts `C/M = 2/3` per residue — not even an integer, and it matches no
cell. The reason is structural: the residue of a mass-`W` subset is `sum p[i] mod M` taken *only over
subsets that also hit mass W*, and the mass constraint correlates which shards, hence which phases, can
co-occur. There is no residue-shifting symmetry that preserves mass, so no balance to appeal to. The
closed form is dead; the two dimensions are coupled and must be carried together.

So I carry both in one table: `dp[m][r]` = number of subsets of the shards seen so far with total mass
exactly `m` and total phase `≡ r (mod M)`. Masses above `W` are useless — a partial weld that overshoots
`W` can never come back down — so I cap the mass axis at `W`, giving a `(W+1) × M` table. Base case
before any shard: only the empty subset, so `dp[0][0] = 1`, everything else `0`. For shard `i` with mass
`wi` and reduced phase `pi = p[i] mod M`, a subset either omits it (value unchanged) or includes it,
mapping a mass-`(m - wi)`, residue-`r` subset to mass `m`, residue `(r + pi) mod M`:

    for m = W down to wi:
        for r in 0..M-1:
            dp[m][(r + pi) mod M] += dp[m - wi][r]   (mod 1e9+7)

The mass loop must descend, and this is where a 0/1 knapsack silently becomes an unbounded one. When I
update `dp[m]` from `dp[m - wi]` while descending, the source row `dp[m - wi]` has not yet been touched
by shard `i` this pass. Ascend instead and the source row has already absorbed shard `i`, so I weld the
same physical shard twice. The failure is concrete: two identical unit shards `(1,0),(1,0)`, `W=2`,
`M=1`, whose only resonant subset is `{both}` (answer `1`). Ascending, shard 1 bumps `dp[1][0]` to `2`
and then `m=2` reads that just-updated `2` into `dp[2][0]`, returning `3` — a shard used twice.
Descending keeps `dp[1][0]` at its pre-shard-1 value when `m=2` reads it, and `dp[2][0]` lands at `1`.
Downward it is.

The residue arithmetic is the other place this problem sets a trap, and it comes straight from `p[i] <=
10^9`. I want the wrap `(r + pi) mod M` to be a single conditional subtraction, `nph = ph + pi; if (nph
>= M) nph -= M;`. That is valid *only* when `pi < M`: then `ph + pi < 2M` and one subtraction lands back
in `[0, M)`. If I skip the reduction and feed a raw `pi` up to `10^9`, `ph + pi` is enormous, one
subtraction leaves it far above `M`, and `dp[m][nph]` writes past the end of an `M`-wide row — a heap
overflow, or silent corruption where it happens to stay in bounds. So `pi = p[i] % M` must happen before
the loop, not after — reduced first, the phase axis is sound up to `10^9`.

The corner cases all fall out of the recurrence with no special-casing.
`n=0`: the shard loop never runs, the table stays `dp[0][0]=1`, and the answer `dp[W][q]` is `1` iff
`W=0` and `q=0`, else `0` — the bare mount has mass and phase both zero. `M=1`: the only residue is `0`
and `q` is forced to `0` by the contract, so every phase reduces away and the count collapses to
"subsets of mass exactly `W`". A shard with `w[i] > W`: the `if (wi > Wcap) continue;` skips its update,
which is correct because skipping means never welding it — its "not taken" branch is already carried in
the existing cells, and that is its only feasible role. An unreachable `(W, q)`: the cell is `0`, printed
directly. Modulus: every cell stays in `[0, MOD)` because I reduce after each add, and the largest
pre-reduction intermediate is `< 2*MOD`, comfortably inside `long long`.

Cost: the table is `(W+1) * M <= 2001 * 200 ≈ 4·10^5` longs (~3.2 MB), and the work is `O(n * W * M) <=
200 * 2000 * 200 = 8·10^7` inner steps — well under the 1 s limit (about 0.04 s on a full-size case),
and the `if (v == 0) continue;` skip over empty source cells only helps. That is what I ship: one
self-contained C++ file, the coupled `O(n*W*M)` counting knapsack with the mass loop descending and the
phase reduced before the single-subtraction wrap. The full program is in the answer.
