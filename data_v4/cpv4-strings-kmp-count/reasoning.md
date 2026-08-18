**What the adversarial input is.** For every length `L` I must report `c[L]`, how many times the length-`L` prefix `S[0..L-1]` occurs as a substring of `S`, overlaps allowed. The input that decides the whole approach is `S = "aaaa...a"` at `n = 2*10^5`. Comparing each of the `n` prefixes against every start position there is `~n^2/2 = 2*10^10` character comparisons — dead under one second. But the obvious "fix," building the KMP failure function and then walking the border chain down from each end position, is *also* `O(n^2)` on exactly that string, because the chain at position `i` is `i` links long. So the real content of this problem is not "use KMP" — it is turning that per-position chain walk into a single linear pass without losing or duplicating a count. I keep two invariants as anchors: `c[L] >= 1` always (the prefix matches itself at position `0`) and `c[n] = 1` always. Counts never exceed `n`, comfortably inside `int`, but I accumulate in `long long` since the counting pass does repeated `+=` into the same cell and the cost is nil.

**The occurrence structure.** `pi[i]` is the length of the longest proper prefix of `S[0..i]` that is also a suffix of `S[0..i]`, with `pi[0] = 0`, built in the standard linear way: `j = pi[i-1]`; while `j > 0` and `S[i] != S[j]`, `j = pi[j-1]`; if `S[i] == S[j]` then `j++`; `pi[i] = j`. The fact I need: a length-`L` prefix occurs *ending at* position `i` exactly when `L` is a border length of `S[0..i]` — i.e. `L` appears in the chain `i+1, pi[i], pi[pi[i]-1], ..., 0`. The top term `i+1` is the prefix `S[0..i]` occurring as itself; the rest, `pi[i]` and below, are the *proper* prefix-suffixes ending at `i`. So counting every entry of the proper chain `pi[i], pi[pi[i]-1], ...` over all `i` gives every non-self occurrence of every prefix, and the self-occurrence — the single match at position `0` — I add as `+1` per length at the end.

**Collapsing the chain walk into one push pass.** Instead of walking each chain, I seed only its top: for each `i` with `pi[i] > 0`, do `cnt[pi[i]]++`. Then I exploit the recursion — every occurrence of the length-`L` prefix is a string whose own longest border is `pi[L-1]`, occurring at the same ending positions, so whatever accumulates in `cnt[L]` must also flow into `cnt[pi[L-1]]`:

```
for each i:            if pi[i] > 0    then cnt[pi[i]] += 1        // seed top of proper chain
for L = n down to 1:   if pi[L-1] > 0  then cnt[pi[L-1]] += cnt[L] // push down the chain
```

Because a proper border is strictly shorter (`pi[L-1] < L`), running `L` from high to low guarantees `cnt[L]` is finished before it donates. The two guards keep counts out of the length-`0` cell, which is never a valid border target and never printed; a push into it would silently swallow counts that should have stopped.

**The bug (first one): the self term must be added before propagation, not after.** Let me see where the extra count at length 4 came from. The length-4 prefix `"abac"` occurs only at position `0`; it has no proper border (`pi[3]=0`). So `cnt[4]` from seeds+push is `0`, and `+1` gives `1` — that part is fine. Re-checking my arithmetic: in the push pass I wrote `cnt[2]=1` after `L=6`, but the *seed* already had `cnt[2]=1`, and `L=6` did `cnt[2]+=cnt[6]=0`, leaving `1` — fine. The real slip is at length... let me recompute `cnt[4]`: seeds gave `cnt[4]=0`; no push targets length 4 except a push *from* some `L` with `pi[L-1]=4`, and no such `L` exists here, so `cnt[4]=0`, then `+1 = 1`. So actually `c[4]=1`. I miscopied above; redoing the vector after the push pass: `cnt[1]=3, cnt[2]=1, cnt[3]=1, cnt[4]=0, cnt[5..7]=0`, then `+1` each gives `4 2 2 1 1 1 1`. That is correct.

So the derivation is right and I made a transcription error in the trace itself — good that I recomputed. But this scare flags the genuinely dangerous variant of the same idea, which I now test: **what if I add the `+1` self term *before* the push pass instead of after?** That is a real, tempting ordering ("a prefix occurs once as itself, so initialize each cnt to 1, then propagate"). Let me trace that broken order on `"abacaba"`. Initialize `cnt[L]=1` for all `L=1..7`, then add seeds: `cnt[1]=1+2=3, cnt[2]=1+1=2, cnt[3]=1+1=2, cnt[4..7]=1`. Push `L=7..1`:
- `L=7,b=3: cnt[3]+=cnt[7]=1 -> 3`.
- `L=6,b=2: cnt[2]+=cnt[6]=1 -> 3`.
- `L=5,b=1: cnt[1]+=cnt[5]=1 -> 4`.
- `L=4,b=0: skip`.
- `L=3,b=1: cnt[1]+=cnt[3]=3 -> 7`.
- `L=2,b=0: skip`.
- `L=1,b=0: skip`.
Result `cnt[1]=7, cnt[2]=3, cnt[3]=3, ...` — wildly wrong (expected `4 2 2`). The defect is precise: the self `+1` placed at every length gets *propagated down the border chain along with the real counts*, so each prefix's self-occurrence is double-, triple-counted into its borders. The self term is an occurrence at position `0` that does **not** recurse into shorter prefixes the way ended-occurrences do, so it must be added strictly *after* all propagation. Conclusion locked in: seed -> push high-to-low -> then `+1`. My final code adds the `+1` last, which the trace now confirms is the only safe order.

**Second debug episode: the propagation direction.** The other classic way to wreck this is to run the push pass *low to high* instead of high to low. I want to feel that failure concretely, so I trace `"aaaa"`, expected `4 3 2 1`. `pi` for `a a a a` is `[0,1,2,3]`. Seed pass (`pi[i]>0`): `i=1 -> cnt[1]++`, `i=2 -> cnt[2]++`, `i=3 -> cnt[3]++`. So `cnt=[_,1,1,1,0]`.

First the *correct* high-to-low push, `b=pi[L-1]`:
- `L=4,b=pi[3]=3: cnt[3]+=cnt[4]=0 -> 1`.
- `L=3,b=pi[2]=2: cnt[2]+=cnt[3]=1 -> 2`.
- `L=2,b=pi[1]=1: cnt[1]+=cnt[2]=2 -> 3`.
- `L=1,b=pi[0]=0: skip`.
`cnt=[_,3,2,1,0]`, then `+1` -> `4 3 2 1`. Correct.

Trace `"aabaaab"` (a tricky one), expected `5 3 2 1 1 1 1`. `S = a a b a a a b`. Compute `pi`: `pi[0]=0`; `i=1 (a==a) pi=1`; `i=2 (b vs S[1]=a no, j=pi[0]=0, b vs a no) pi=0`; `i=3 (a vs S[0]=a yes) pi=1`; `i=4 (a, j=pi[3]=1, a vs S[1]=a yes) pi=2`; `i=5 (a, j=pi[4]=2, a vs S[2]=b no -> j=pi[1]=1, a vs S[1]=a yes) pi=2`; `i=6 (b, j=pi[5]=2, b vs S[2]=b yes) pi=3`. So `pi=[0,1,0,1,2,2,3]`. Seed (`pi[i]>0`): `i=1->cnt[1]++`, `i=3->cnt[1]++`, `i=4->cnt[2]++`, `i=5->cnt[2]++`, `i=6->cnt[3]++`. `cnt=[_,2,2,1,0,0,0,0]`. Push `L=7..1`, `b=pi[L-1]`:
- `L=7,b=pi[6]=3: cnt[3]+=cnt[7]=0 -> 1`.
- `L=6,b=pi[5]=2: cnt[2]+=cnt[6]=0 -> 2`.
- `L=5,b=pi[4]=2: cnt[2]+=cnt[5]=0 -> 2`.
- `L=4,b=pi[3]=1: cnt[1]+=cnt[4]=0 -> 2`.
- `L=3,b=pi[2]=0: skip`.
- `L=2,b=pi[1]=1: cnt[1]+=cnt[2]=2 -> 4`.
- `L=1,b=pi[0]=0: skip`.
`cnt=[_,4,2,1,0,0,0,0]`, then `+1` -> `5 3 2 1 1 1 1`. Matches expected exactly. The order (seed, high-to-low push, late `+1`) reproduces the known answer.

**Two ordering traps.** The idea is easy; the bookkeeping has two ways to go wrong, and each one changes the output.

First, the self `+1` must be added strictly *after* the push, not folded in as an initialization. The tempting move is "each prefix occurs once as itself, so start every `cnt[L]` at `1` and then propagate." That is wrong: the self-occurrence sits at position `0` and does not recurse into shorter prefixes the way ended-occurrences do, so if it is present during the push it rides down every border chain and gets re-counted. On `"abacaba"` that inflates `c[1]` from `4` to `7`. The self term goes in last.

Second, the push must run high-to-low, not low-to-high. Reversed, when I process `L` I read `cnt[L]` before the donations that later flow *into* it from longer lengths have arrived, so those contributions never reach `cnt[L]`'s own border. On `"aaaa"` low-to-high yields `3 3 2 1` instead of `4 3 2 1` — a lost count, not a doubled one. High-to-low is the only order where each cell is complete at the moment it donates.

On `"abacaba"` (`pi = [0,0,1,0,1,2,3]`) the seeds land at `cnt[1]=2, cnt[2]=1, cnt[3]=1` (from `i=2,4` / `i=5` / `i=6`). Pushing down `b=pi[L-1]` from `L=n`, the only donation with nonzero content is `L=3, b=1`, sending `cnt[3]=1` into `cnt[1]` to make it `3`; every longer prefix here occurs solely at position `0`, so the higher cells are empty and donate nothing. The self `+1` per length then produces `4 2 2 1 1 1 1`. This is where the high-to-low order earns its keep: `cnt[3]` is final before `L=3` reads it.

**Edge behavior.** `n = 1` (`S="a"`): no seed, no push, `+1` gives `1`, and `c[n]=1` holds. All-distinct `"abcde"` has `pi` all zero, so seeds and pushes do nothing and every count is the bare `+1`: `1 1 1 1 1`. Maximal overlap `"a"*n` gives `c[L] = n - L + 1` (the length-`L` block starts at `0..n-L`); the push chain reconstructs exactly that. The output is `cnt[1..n]` space-separated with a trailing newline, and the framework's `if (!(cin >> s)) return 0;` covers absent input though the constraints promise `n >= 1`.

**Cost.** The failure function is amortized `O(n)` and each of the three passes is a single loop — no border chain is ever walked explicitly, which is what keeps `"a"*n` linear. Linear time and space; the full three-pass program is in the answer.
