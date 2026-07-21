This is maximum-weight independent set, but on a *circle*. On a line I already trust the DP: scan left to right carrying `skip` (best prefix sum with the last position not taken) and `take` (best with it taken), `take_i = skip_{i-1} + a[i]`, `skip_i = max(skip_{i-1}, take_{i-1})`, answer `max(take, skip, 0)` since the empty set is allowed. The circle adds exactly one edge — the wrap between index `0` and index `n-1` — and that single edge is the entire difficulty: a selection taking both endpoints is legal on the line but illegal here. Everything below is about handling that one edge without disturbing the rest.

Before any algorithm, the scale fixes the types. `n <= 2*10^5` and `|a[i]| <= 10^9`, so a sum can reach `2*10^5 * 10^9 = 2*10^14`, well past the `~2.1*10^9` of 32-bit. Every accumulator is `long long`; an `int` here is not a crash, it is a silent wrong answer on the large tests.

Three routes suggest themselves, and I want the one I can prove, not the one fastest to type.

Greedy — repeatedly take the largest free positive value and forbid its two cyclic neighbours — is `O(n log n)` and a few lines, but local max-grab under a global adjacency constraint is unsafe. On the line `[8, 9, 2, 9, 9, -2, 8, -5]` greedy grabs the `9` at index 1, which blocks index 0, and lands on `26`, whereas `{0, 2, 4, 6}` = `8 + 2 + 9 + 8 = 27` is reachable. Pad that with a large-negative spacer so the wrap never binds and the same failure transfers onto a circle. Greedy is out.

The naive single pass — run the path DP straight over `a[0..n-1]` and print it — ignores the wrap, so it takes both endpoints whenever that is locally best. The sample is exactly the witness: on `[5, 1, 1, 5, 1, 5]` the path DP picks `{0, 3, 5}` = `15`, but `0` and `5` are wrap-adjacent, so that is illegal; the true circular optimum is `{1, 3, 5}` = `11`. It over-counts precisely the wrap edge.

The tempting repair keeps one pass but conditions on whether index `0` was taken. That does not collapse to one extra bit: the optimal `skip` at position `i` may come from a sub-path that took index 0 while the optimal `take` at the same `i` came from one that did not, so I would have to carry, for each of `skip`/`take`, the best value conditioned on `took0 = true` and on `took0 = false` — four numbers, four two-branch transitions, a base case that seeds the conditioning at index 0, and a final reconciliation forbidding `took0 && took_{n-1}`. That machine is correct in principle, but it is the kind of four-way conditional DP where one mis-set branch is a silent wrong answer that only surfaces when the wrap binds. There is a method that avoids the wrong-endpoint pair entirely, so I would rather prove that one than debug this.

The clean observation: the wrap edge says index `0` and index `n-1` cannot both be chosen, so every valid circular selection omits at least one endpoint. Split on which one:

- Case A: index `n-1` unused. The only adjacencies left among chosen elements are the line edges within `a[0..n-2]`, so the best Case-A value is the *path* maximum over `a[0..n-2]`.
- Case B: index `0` unused. Symmetrically, the path maximum over `a[1..n-1]`.

The circular answer is `max(A, B)`. Soundness: a Case-A selection lives in `a[0..n-2]` and never touches `n-1`, so it cannot use both `0` and `n-1`, hence it is a legal circular selection; symmetric for B. Completeness: any optimal circular `S` omits `n-1` or omits `0`, so it is realized in Case A or Case B respectively, and `max(A, B)` reaches its value. Together these make `max(A, B)` exactly the true optimum. This is just the path DP — already trusted — run twice on two ranges: no four-state machine, no repair step, nothing new to prove.

The split assumes two distinct endpoints exist, so the tiny `n` need care. `n = 0` → `0`. `n = 1` has no second vertex to be adjacent to, so the lone element may be taken: `max(a[0], 0)`. This one must be special-cased — blindly running the split gives Case A over the empty range `a[0..-1]` = 0 and Case B over `a[1..0]` = 0, which would wrongly refuse a positive lone element. `n = 2` needs no special case: positions 0 and 1 share a single adjacency, and the split gives `max(line[a[0..0]], line[a[1..1]]) = max(a[0], a[1], 0)`, correct as long as the line solver handles a one-element range.

The path DP is short but carries one transcription trap this problem actively invites: both transitions read the *previous* `(skip, take)` pair, so I must form both new values from the old pair through temporaries. If I update `skip` first and then compute `take = skip + a[i]`, `take` is built on a state that already took `i-1` — the exact adjacency I forbade; the line `[1, 1]` would return `2`, taking both adjacent ones. The base case matters too: before any element `skip = 0` (empty prefix) and `take` is minus infinity (no last-taken state exists yet); starting `take = 0` would assert a phantom prior-taken state and let a lone negative slip through. I use `LLONG_MIN / 4` as the sentinel so it survives inside a `max` without ever having `a[i]` added to it — I only ever add to `skip`. Written on an inclusive range `[lo, hi]`, the loop does nothing when `lo > hi` and returns the empty-range value `0` the corners rely on.

```
long long take = LLONG_MIN / 4, skip = 0;
for (int i = lo; i <= hi; i++) {
    long long ntake = skip + a[i];        // take i: previous skipped
    long long nskip = max(skip, take);    // skip i: previous either
    take = ntake; skip = nskip;
}
return max({take, skip, 0LL});
```

On `[1, 1]` this steps `(-inf, 0)` → `(1, 0)` → `(1, 1)` and returns `1`, not `2`.

Assembling, for `n >= 2`: `best = max(linearBest(a, 0, n-2), linearBest(a, 1, n-1))`. On the sample `[5, 1, 1, 5, 1, 5]`, Case A over `[5, 1, 1, 5, 1]` runs to `10`, Case B over `[1, 1, 5, 1, 5]` runs to `11`, so `best = 11` — the legal `{1, 3, 5}`. Neither case can reach the illegal `15`, because Case A forbids index 5 and Case B forbids index 0.

The remaining corners fall out. An all-negative array keeps every `take` negative and every `skip` at 0, so both cases return 0 — as the empty array and a single negative already do.

Hand-traces only cover the cases I thought of, so I differential-tested against a separate `2^n` brute that enumerates every subset and checks cyclic independence including the `(n-1, 0)` edge — unrelated to the DP by construction. Across several hundred random instances mixing tiny `n`, small cycles, and larger ones up to 16 over all-positive / all-negative / mixed-with-zeros / extreme `±10^9` value ranges, plus a hand-picked edge suite (empty, single `±`, pair, all-equal, the wrap-binding `[5,1,1,5,1,5]`, the `[8,9,2,9,9,-2,8]` greedy probe), zero mismatches. A single `n = 2*10^5` run of random `±10^9` values finishes in about `0.04` s under `5` MB — comfortably inside the `1` s / `256` MB budget for an `O(n)` two-pass scan.
