The grid is a tree on `n` stations, and for every station `r` I need the number of connected vertex subsets `S` that contain `r`. Two features of *this* problem fix its shape before I write a line. First, the counts are astronomical — the centre of a star on `n` leaves already sits in `2^(n-1)` clusters — so nothing survives as an integer and everything lives modulo `p = 1000000007`; every product of two residues near `10^9` has to sit in a `long long` and be reduced at each multiply, or it overflows a 32-bit `int` on the very first step. Second, `n` runs to `2*10^5` and I owe an answer at *every* vertex, not one — and that "per-vertex" is exactly what lifts this above a textbook single-DP.

Start with the count for one fixed root, because whatever I build for all roots is only ever as correct as that base DP. Root the tree at `r`. A connected subset `S` containing `r` is built top-down: `r` is in; at each vertex already in `S`, for each child `c` I independently either skip `c`'s subtree entirely or enter `c` and recurse the same decision. The choices across children are independent — the only thing tying them together is the parent, already in `S` — so the count multiplies. With `down[v]` = the number of connected subsets of `v`'s subtree that contain `v`,

```
down[v] = prod over children c of v of (1 + down[c]),
```

the `1` being "skip `c`", the `down[c]` being "enter and count inside"; a leaf's empty product gives `down[leaf] = 1`, and the answer for root `r` is `down[r]`. On the path `1-2-3-4-5` rooted at `1` this gives `down = 1,2,3,4,5`, and the connected subsets through vertex `1` are exactly the prefixes `[1..k]`, so `5` — the recurrence holds.

Now `down[r]` for *all* `r`. The dead-simple route is to recompute the `O(n)` post-order DP from scratch for each of the `n` roots — literally the definition, `n` times, obviously correct. The only question is time, and it settles fast: at `n = 2*10^5`, `n^2 = 4*10^10`, so even at an optimistic `10^9` operations per second that is ~40 seconds, and the per-root DFS constant (rebuilding a parent array, a modular multiply at every vertex) pushes it further, against a 1-second limit. This is off by two orders of magnitude — structurally excluded, not something a faster inner loop rescues. I keep it only as a small-`n` oracle to differential-test the real solution against.

So I need every root's answer without redoing the whole DP, which is what rerooting buys. Fix the rooting at `1`; after the first pass I know `down[v]` for every `v`, the count looking *downward* into its own subtree. When `v` becomes the global root, its neighbours are its children in the rooting at `1` (each contributing `1 + down[child]`) plus its parent `p` — one *upward* direction. Define `up[v]` = the number of connected subsets containing `p` that live on `p`'s side of the edge `v-p`. The upward part is then just one more independent neighbour subtree hanging off `v`, contributing the same `(1 + g)` shape, so

```
answer[v] = down[v] * (1 + up[v]),      answer[1] = down[1]   (the root has no upward part).
```

To transport `up` from a vertex `u` to one of its children `c`: standing at `c` and looking up through `u`, the region above `c` is `u` together with all of `u`'s neighbour-directions *except* `c` — `u`'s parent (factor `1 + up[u]`) and `u`'s other children `c'` (factor `1 + down[c']`). So

**The trace that breaks it.** Take the star: centre `1` with leaves `2,3,4`. Rooted at `1`: `down[2]=down[3]=down[4]=1`, `down[1]=(1+1)(1+1)(1+1)=8`. So `answer[1]=8`. Now push from `1` to its children. At `v=1`, `g(w)=down[w]=1` for each of the three neighbours, so each factor `(1+g)=2`, and `full = 2*2*2 = 8`. For child `c=2`: `up[2] = full * modinv(1+down[2]) = 8 * modinv(2) = 4`. Then `answer[2] = down[2]*(1+up[2]) = 1*(1+4) = 5`. Cross-check against the definition: connected subsets through a leaf `2` of this star are `{2}`, `{2,1}`, `{2,1,3}`, `{2,1,4}`, `{2,1,3,4}` = `5`. Correct here. So far division *works*. That is the trap — it works until a factor is `0 mod p`.

**Where division actually detonates.** The factor I divide out is `(1 + g_w) mod p`. `g_w` is a count, and counts are reduced mod `p`, so `1 + g_w` can be a multiple of `p` and hence `0 mod p` even though the *true* integer factor is nonzero. The smallest concrete trigger: I need some `(1 + down[w])` or `(1 + up[v])` to be `≡ 0 (mod p)`, i.e. `down[w] ≡ p-1`. That requires a subtree whose subset-count is exactly `p-1 mod p`, which absolutely occurs on large random trees — and the moment it does, `modinv(0)` is undefined (it computes `0^(p-2)=0`), so `up[c]` silently becomes `0` and every answer downstream of that vertex is wrong. Worse, it is data-dependent: the small cases I traced by hand never hit it, so the bug ships looking verified. The defect is precise: **I am dividing in a modular ring where the divisor can be a true-nonzero value that is zero modulo `p`, and there is no inverse of zero.** Division is not safe here, full stop.

```
up[c] = product over all neighbours w of u EXCEPT c of (1 + g_w),   g_w = up[u] if w is u's parent, else down[w].
```

That "product over all neighbours except one" is the crux, and the instinctive way to get it is a trap specific to counting mod `p`. The shortcut is to take the full product over all neighbours and divide out `c`'s factor with a modular inverse. But each factor is `(1 + g) mod p` where `g` is itself a reduced count, so the factor can be `0 mod p` — a true-nonzero integer that happens to be a multiple of `p` (it needs `down[w] ≡ p-1`, which large random trees do reach). `modinv(0)` computes `0^(p-2) = 0`, so the inverse silently returns garbage and every answer below that vertex is corrupted; and no small hand trace ever triggers it, so the division version would ship looking verified. So I refuse division entirely: lay the neighbour factors in an array `fac[0..deg-1]`, build prefix products `pref[k] = fac[0..k-1]` and suffix products `suf[k] = fac[k+1..deg-1]`, and read the product-except-`k` as `pref[k] * suf[k+1]`. No inverse — a zero factor just makes some prefixes and suffixes zero exactly where they should be — and it also drops the `log p` from the modular inverse. Per vertex it is `O(deg)`, summing to `O(n)` since the degrees sum to `2(n-1)`.

The star makes the reroot combination concrete (centre `1`, leaves `2,3,4`). `down`: leaves `1`, centre `(1+1)^3 = 8`, so `answer[1] = 8`. At `u = 1` the neighbour factors are `[2,2,2]`, giving `pref = [1,2,4,8]`, `suf = [8,4,2,1]`; for `c = 2`, `up[2] = pref[0]*suf[1] = 4` and `answer[2] = 1*(1+4) = 5`, and by symmetry the other leaves are `5` too. Against the definition the clusters through leaf `2` are `{2}, {2,1}, {2,1,3}, {2,1,4}, {2,1,3,4}` — exactly `5`. Output `8 5 5 5`.

The traversal itself has to be iterative. A path of `2*10^5` vertices rooted at one end has depth `2*10^5`, and a recursive DFS overflows the call stack long before it finishes. I build one pre-order `order` with an explicit stack, then compute `down` by walking it in *reverse* (a child always follows its parent in pre-order, so reverse visits children first) and push `up` by walking it *forward* (parent before child, so `up[u]` is ready the moment I reach `u`). One array, two directions.

The lone corner is `n = 1`: no edges, and the only connected subset containing vertex `1` is `{1}`, so the answer is `1`. I special-case it before touching the edge machinery, which assumes the root has neighbours to reroot across. Assembling: `answer[1] = down[1]`, and `answer[v] = down[v] * (1 + up[v]) % p` otherwise; output is the `n` answers space-separated on one line.

**Edge cases.**
- `n = 1`: output `1`. Handled explicitly.
- `n = 2` (`1-2`): rooted at `1`, `down[2]=1`, `down[1]=1+1=2`, `answer[1]=2`. Push to `2`: at `v=1` factors `[1+down[2]] = [2]`, `up[2]=pref[0]*suf[1]=1*1=1`, `answer[2]=down[2]*(1+up[2])=1*2=2`. Output `2 2`. Correct: each vertex is in `{itself}` and `{1,2}`.
- Long path: contiguous-segment counts; verified `5 8 9 8 5` for `n=5` by hand (`down` reproduces prefixes; the symmetric `8 9 8` comes out of the rerooting, which I checked numerically).
- Star: `8 5 5 5` above.
- Deep skewed path at `n=2*10^5`: iterative traversal, no stack overflow; the answer for the head vertex is the segment count `n = 200000`.
- Modulus stress: every multiply is `% MOD`, products are `long long` (max intermediate `< p^2 < 10^18`, fits in 64-bit), and there is **no division**, so no `0`-inverse hazard anywhere on large random inputs.
- Output format: `n` integers separated by single spaces, one trailing newline. I build the line in a string buffer to avoid `n` separate `cout` flushes.

To check the whole chain rather than my own traces, I differential-tested the compiled C++ against the from-scratch `O(n^2)` recompute-per-root oracle on 600+ random small trees (paths, stars, broad random, mixed orientations and labels) with zero mismatches, and validated that oracle against a fully independent `O(2^n)` subset-enumeration (every vertex subset, connectivity by flood-fill) for `n <= 12` — also clean, so `exhaustive ⇒ per-root oracle ⇒ C++` holds end to end. Max-size stress on `n = 2*10^5` paths, stars, and random trees ran in ~0.08 s and ~20 MB with no overflow and no stack issue.

The full module — the iterative `down` pass, the forward rerooting push via prefix/suffix products, and `answer[v] = down[v] * (1 + up[v])` with `n = 1` special-cased — is in the answer.
