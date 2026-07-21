One number fixes the data types before any algorithm does: `n` up to `2*10^5`, each `|a[i]|` up to `10^9`, so a selected sum can reach `2*10^14` — roughly ten thousand times past the `~2.1*10^9` ceiling of a 32-bit `int`. Every accumulator here has to be `long long`. An `int` would be correct on every small test and silently wrong only on the large ones, which is the worst failure to chase. With that pinned: I must pick a subset of positions with no two adjacent, maximize the sum, and since the empty set is legal the answer never drops below `0`.

Two routes are on the table. Greedy — repeatedly grab the largest free positive value and block its two neighbours — is `O(n log n)` and three lines, but adjacency is a global constraint while greedy commits locally, exactly the shape where local grabs go wrong. So I try to break it. Take `a = [8, 9, 2, 9, 9, -2, 8, -5]`. Greedy takes a `9` at index 1 (blocking 0 and 2), the `9` at index 3 (blocking 2 and 4), then the `8` at index 6: total `9 + 9 + 8 = 26`. But indices `0, 2, 4, 6` are pairwise non-adjacent and give `8 + 2 + 9 + 8 = 27`. Snatching the big `9` at index 1 blocked index 0, and that one block cost more than it saved. Greedy is out.

The linear alternative carries two prefix optima left to right: `skip`, the best sum over `a[0..i]` with position `i` not taken, and `take`, the best with `i` taken. The future only cares whether `i` was taken — that is all that constrains `i+1` — so two values suffice. Skipping `i` leaves `i-1` unconstrained: `skip_i = max(skip_{i-1}, take_{i-1})`. Taking `i` forces `i-1` untaken and adds `a[i]`: `take_i = skip_{i-1} + a[i]`. Before any element, `skip = 0` (the empty selection) and `take = -inf` (no last-taken state exists yet). The answer is `max(take_{n-1}, skip_{n-1}, 0)`, the trailing `0` being the empty selection that floors all-negative inputs at `0`.

Transcribing this is where it goes wrong. The tempting in-place loop is

```
long long take = 0, skip = 0;
for (int i = 0; i < n; i++) {
    skip = max(skip, take);
    take = skip + a[i];
}
```

and it destroys `skip` before `take` reads it. Trace `[1, 1]`, whose answer is `1` — the two ones are adjacent, I keep one. i=0: `skip = max(0,0) = 0`, `take = 0 + 1 = 1`. i=1: `skip = max(0,1) = 1`, then `take = skip + a[1] = 1 + 1 = 2`. It reports `2`, taking both adjacent ones. The `skip` I fed into `take` on `i=1` already folded in index 0's `take`, so the new `take` builds on a state that took `i-1` — the adjacency I forbade. Both transitions need the *previous* pair, so I compute both new values into temporaries and only then commit. Initializing `take = 0` is a second trap: it asserts a last-taken state of sum `0` before any element, which would let a lone negative slip through as if it were free; `take` must start at negative infinity.

With temporaries and a proper sentinel:

```
long long take = LLONG_MIN / 4, skip = 0;
for (int i = 0; i < n; i++) {
    long long ntake = skip + a[i];
    long long nskip = max(skip, take);
    take = ntake; skip = nskip;
}
```

Re-trace `[1, 1]`: start `(-inf, 0)` -> i=0 `(1, 0)` -> i=1 `ntake = 0+1 = 1`, `nskip = max(0,1) = 1` -> `(1, 1)`, answer `max(1, 1, 0) = 1`.

The corners fall straight out. `n = 0`: the loop never runs, `max(-inf, 0, 0) = 0`. A lone negative `[-7]`: `ntake = -7`, `nskip = 0`, answer `max(-7, 0, 0) = 0` — take nothing rather than a loss. All-negative: every `take` stays negative, `skip` stays `0`, answer `0`. And the sentinel is safe: `LLONG_MIN/4` is only ever read inside a `max`, never has `a[i]` added to it (I only add to `skip`), so it cannot underflow.

So I ship the `O(n)`, `O(1)`-space DP: 64-bit accumulators against the overflow, both transitions reading the previous pair through temporaries, and `take` seeded at `-inf`.
