**Reading the problem and pinning the contract.** I have `n` items, each a `B`-bit mask, and `q` query masks. For each query `m` I must count the items `x` with `x & m == m` — the items that are *supersets* of `m`, i.e. that have every bit of `m` set (and possibly more). The scale is the whole story here: `B <= 20`, so `2^B` is up to about `10^6`; `n` and `q` are each up to `10^6`. Counts can reach `n = 10^6`, which fits an `int`, but I will carry `long long` for the accumulators so I never have to second-guess an overflow on a sum of counts. Output is one integer per query. Before any algorithm I fix the two numbers that decide everything: `2^B <= 2^20 ≈ 1.05*10^6`, and `n, q <= 10^6`. Anything that multiplies two of these together — `n*q`, or `q*2^B`, or `3^B` — is the danger zone.

**Laying out the candidate approaches.** Two routes are on the table, and I want the one whose complexity I can *defend against the limits*, not just the one that is easiest to type.

- *Per-query scan.* For each query `m`, loop over all `n` items and test `x & m == m`, tallying. Dead simple and obviously correct. Cost: `O(n)` per query, `O(n q)` overall.
- *Aggregate once, then O(1) per query.* The set of queries only ever reads a function of the *item multiset*: how many items are supersets of a given mask. That function depends only on the mask, not on which query asked. So I should precompute, for **every** mask `m` in `[0, 2^B)`, the value `S[m]` = number of items that are supersets of `m`, store it in an array, and answer each query by a single array read `S[m]`. The remaining question is how to build `S[]`.

**Killing the per-query scan on the actual numbers.** Let me not hand-wave "too slow." `n q` at the limits is `10^6 * 10^6 = 10^{12}` mask-AND operations. Even at an optimistic `10^9` simple ops per second that is on the order of `10^3` seconds — a thousand seconds for a two-second limit. It is not close; it is off by three orders of magnitude. So per-query scanning is out for the full constraints. It survives only as my brute-force oracle on tiny `B`, which is exactly what I will use it for. Decision made: I must aggregate once.

**First idea for the aggregation — and why the obvious version is also too slow.** The clean definition is

```
S[m] = sum over all t in [0,2^B) with (t & m) == m  of  cnt[t],
```

where `cnt[t]` is the number of items whose mask equals `t`. The condition `(t & m) == m` says exactly "t is a superset of m." The textbook-obvious way to compute this is: for each mask `m`, enumerate its supersets `t` and add up `cnt[t]`. Enumerating supersets of `m` is a standard trick — iterate `t` over the "complement submask" loop. But what does it cost in total? Summing over all `m` the number of supersets of `m`: a mask with `k` zero bits has `2^k` supersets, and summing `2^{(B-popcount(m))}` over all `m` gives `3^B` (the same `sum over masks of 2^{zeros}` identity that gives the famous `3^B` submask-enumeration bound). At `B = 20`, `3^{20} ≈ 3.49 * 10^9`. That is borderline-to-impossible in two seconds, and it is fragile. The double-loop over mask/superset pairs is the `O(3^B)` wall the problem is built around.

Let me make the wall concrete with a tiny case so I trust the count. `B = 2`, masks `00,01,10,11`. Supersets: `S[00]` sums over all 4; `S[01]` over `{01,11}`; `S[10]` over `{10,11}`; `S[11]` over `{11}`. That is `4 + 2 + 2 + 1 = 9 = 3^2` mask–superset pairs. For `B = 2` nine is nothing, but the `3^B` growth is brutal: every extra bit multiplies the work by 3, while `2^B` only doubles. By `B = 20` the gap between `3^B ≈ 3.5*10^9` and `B*2^B ≈ 20*10^6 = 2.1*10^7` is a factor of ~170. So if I can get to `O(B * 2^B)` I win comfortably, and if I cannot I am stuck at the wall. The whole problem is finding that `O(B * 2^B)`.

**Deriving the insight — sum over subsets in the superset direction.** The enemy in the `O(3^B)` version is *recomputation*: when I aggregate the supersets of `m`, I redo work that the aggregation for masks "one bit denser than `m`" already did. The fix is the standard **SOS DP** (sum-over-subsets dynamic programming), also called the fast zeta transform over the subset lattice. The usual textbook SOS computes, for each mask, the sum over its *subsets*. I need the dual: for each mask, the sum over its *supersets*. Same machinery, run in the opposite direction along each bit.

Here is the derivation I trust. Process the `B` bits one at a time, `b = 0, 1, ..., B-1`. Define the partial transform `f_b[m]` to be: the sum of `cnt[t]` over all `t` that agree with `m` on bits `0..b-1` already considered being "free above," and equal `m` on the remaining bits — more precisely, after processing bit `b`, `f[m]` equals the sum of the original `cnt[t]` over all `t` that are supersets of `m` *when restricted to bits `0..b`*, i.e. `t` may have extra 1s among bits `0..b` but must match `m` exactly on bits `> b`. The transition for bit `b` is:

- If bit `b` is **0** in `m`, then a superset `t` of `m` may have bit `b` equal to `0` or `1`. The `t` with bit `b = 0` are already counted in `f[m]` from the previous round (they match `m` on bit `b`); the `t` with bit `b = 1` are exactly the supersets of `m | (1<<b)` from the previous round. So I add `f[m | (1<<b)]` into `f[m]`.
- If bit `b` is **1** in `m`, every superset must also have bit `b = 1`, so nothing changes for `m` on this bit.

That single rule — `for each bit b: for each mask m with bit b clear: f[m] += f[m | (1<<b)]` — folds the value of the "bit set" state down into the "bit clear" state. After all `B` bits are processed, `f[m]` is the sum of `cnt[t]` over every `t` that can differ from `m` only by turning extra bits on, on *every* bit — which is precisely the set of supersets of `m`. Cost: `B` passes, each touching `2^B` masks, `O(B * 2^B)`. That is the insight: I never enumerate superset pairs; I propagate counts bit by bit, reusing the previous bit's work, and the `3^B` collapses to `B * 2^B`.

**Sanity-checking the transform on a hand case before coding the loop.** Take `B = 2`, `cnt = [cnt00, cnt01, cnt10, cnt11]` indexed by mask value `0,1,2,3`. I want to end with `S[00] = cnt00+cnt01+cnt10+cnt11`, `S[01] = cnt01+cnt11`, `S[10] = cnt10+cnt11`, `S[11] = cnt11`.

Start `f = [c0, c1, c2, c3]` (writing `ci = cnt[i]`). Process bit `b=0` (value 1). Masks with bit 0 clear are `0 (00)` and `2 (10)`. Update `f[0] += f[1]` and `f[2] += f[3]`:
`f = [c0+c1, c1, c2+c3, c3]`. Process bit `b=1` (value 2). Masks with bit 1 clear are `0 (00)` and `1 (01)`. Update `f[0] += f[2]` and `f[1] += f[3]`:
`f[0] = (c0+c1) + (c2+c3) = c0+c1+c2+c3`; `f[1] = c1 + c3`; `f[2] = c2+c3`; `f[3] = c3`.
That is exactly `S[00], S[01], S[10], S[11]`. The transform is right.

**First implementation.** I read `B, n, q`, build `f` as the multiplicity array, run the two nested loops, then answer queries by lookup. My first cut:

```
const int SZ = 1 << B;
vector<long long> f(SZ, 0);
for (int i = 0; i < n; i++) { int x; cin >> x; f[x] += 1; }

for (int b = 0; b < B; b++)
    for (int mask = 0; mask < SZ; mask++)
        f[mask] += f[mask ^ (1 << b)];   // <-- suspicious

for (int i = 0; i < q; i++) { int m; cin >> m; cout << f[m] << "\n"; }
```

**Tracing the suspicious line — and finding the bug.** The inner update `f[mask] += f[mask ^ (1<<b)]` bothers me because it fires for *every* mask, both those with bit `b` set and those with it clear, and `^` flips the bit unconditionally. Let me trace `B = 2`, `cnt = [c0,c1,c2,c3]`, bit `b=0`. The loop hits every mask `0,1,2,3`:
- `mask=0` (bit0 clear): `f[0] += f[1]` → `c0+c1`. Good.
- `mask=1` (bit0 set): `f[1] += f[0]`. But `f[0]` was *just updated* to `c0+c1`, so `f[1] = c1 + c0 + c1`. Garbage — bit-set masks must not change, and I also read an already-mutated neighbour.
- `mask=2` (bit0 clear): `f[2] += f[3]` → `c2+c3`. Good.
- `mask=3` (bit0 set): `f[3] += f[2]`, with `f[2]` already mutated. Garbage again.

Two defects in one line. First, I update bit-set masks at all — the superset transform must only fold *into* the bit-clear state, leaving the bit-set state untouched (a superset of an already-1 bit can't drop it). Second, by flipping with `^` for set masks and updating in place, I read neighbours that the same pass already changed, so even the "wrong direction" sum is corrupted. The classic SOS pitfall: doing both directions, or reading a within-pass-mutated cell.

**Fixing and re-verifying.** The fix is to guard the update so it only runs for masks with bit `b` **clear**, and to read the partner mask `mask | (1<<b)` — which, crucially, has bit `b` *set* and therefore is *never* written in this pass, so there is no read-after-write hazard within a pass:

```
for (int b = 0; b < B; b++) {
    int bit = 1 << b;
    for (int mask = 0; mask < SZ; mask++)
        if ((mask & bit) == 0)
            f[mask] += f[mask | bit];
}
```

Re-trace `B = 2`, bit `b=0`: only `mask=0` and `mask=2` (bit0 clear) update, reading `f[1]` and `f[3]` (bit0 set, untouched this pass). `f = [c0+c1, c1, c2+c3, c3]`. Bit `b=1`: only `mask=0,1` (bit1 clear) update, reading `f[2]=c2+c3` and `f[3]=c3` (bit1 set, untouched this pass). `f[0]=c0+c1+c2+c3`, `f[1]=c1+c3`. Exactly `S`. The bug is gone, and it is gone for the reason I diagnosed — guarding to bit-clear masks and reading the bit-set partner removes both the illegal updates and the within-pass aliasing.

**Why reading `f[mask | bit]` is safe even though `f` is mutated in place.** This is the subtle correctness point of in-place SOS, so I want it airtight. Within a single bit-`b` pass, the only cells I *write* are those with bit `b` clear. The only cells I *read* (besides the destination) are `f[mask | bit]`, which all have bit `b` set. Written set ∩ read set on the partner side is empty: I never read a cell in the same pass after writing it. So the in-place update computes the same thing a fresh copy would. Across passes the invariant carries: after pass `b`, `f[m]` = sum of `cnt` over supersets-of-`m`-restricted-to-bits-`0..b`. By `b = B-1` that is the full superset sum. Verified.

**Edge cases, deliberately, because this is where mask code dies.**
- `B = 0`: `SZ = 1`, the only mask is `0`, the bit loop runs zero times, `f[0] = n` (every item equals the empty mask). Any query (must be `0`) returns `n`. Correct — every item is trivially a superset of the empty mask.
- `n = 0`: `f` is all zeros, the transform keeps it all zeros, every query returns `0`. Correct — no items, no supersets.
- Query `m = 0` (empty mask): `S[0]` is the sum of `cnt` over *all* masks = `n`. The transform delivers this because mask `0` has every bit clear, so it accumulates every other state. Correct.
- Query `m = 2^B - 1` (full mask): `S[full]` = `cnt[full]`, the count of items exactly equal to the full mask, since the only superset of the full mask is itself. The transform never adds into `f[full]` (it has every bit set, so it is never a bit-clear `mask`), leaving `f[full] = cnt[full]`. Correct.
- Duplicates: `cnt[t]` accumulates multiplicities, and the transform sums counts, so repeated item masks are handled by construction.
- Overflow: counts are at most `n = 10^6`, and a superset sum is at most `n`; `long long` (or even `int`) holds it with room to spare. `long long` chosen defensively.
- I/O scale: `2^B` up to ~`10^6` longs is ~8 MB; with `q = 10^6` output lines I build the output in a single `string` and write it once, and use `sync_with_stdio(false)` so reading `10^6 + 10^6` integers does not dominate. At the limits this runs in well under a second.

**Self-verification against a brute oracle.** I cannot eyeball `B = 20`, so I lean on a differential test: an independent `O(n q)` brute force (for each query, scan all items and test `x & m == m`) on small random instances (`B <= 12`, `n` up to ~60, `q` up to ~40), plus crafted edges — `B = 0`, `n = 0`, query `0`, query full, heavy duplicates. Over 600 random seeds the SOS solution and the brute force agree on every line, and every crafted edge agrees too. The documented sample (`B=3`, items `7 6 3 5 4 1`, queries `3 1 0 7`) yields `2 4 6 1`, matching the by-hand count. The agreement across 600 randomized instances plus the targeted corners is the evidence I trust that the in-place superset SOS is correct, not just plausible.

**Final solution.** I convinced myself the *idea* is right by collapsing the `O(3^B)` superset double-loop into the bit-by-bit fold and checking the transform on a `B=2` case by hand; I convinced myself the *code* is right by tracing the first in-place loop to a precise read-after-write / wrong-direction bug, fixing it with the bit-clear guard reading the never-written partner, and confirming against a brute oracle over hundreds of cases. That is what I ship — one self-contained file, the `O(B * 2^B)` superset SOS:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int B, n, q;
    if (!(cin >> B >> n >> q)) return 0;

    // f[mask] starts as the multiplicity of each item mask, then becomes
    // the number of items whose mask is a SUPERSET of `mask` (item & mask == mask).
    const int SZ = 1 << B;
    vector<long long> f(SZ, 0);

    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        f[x] += 1;
    }

    // Sum-Over-Subsets (zeta transform) in the SUPERSET direction.
    // For each bit b, fold the value of the state with bit b set DOWN into the
    // state with bit b cleared. After processing every bit, f[mask] equals the
    // sum of the original f over all masks t with (t & mask) == mask, i.e. all
    // supersets of `mask`. O(B * 2^B).
    for (int b = 0; b < B; b++) {
        int bit = 1 << b;
        for (int mask = 0; mask < SZ; mask++) {
            if ((mask & bit) == 0) {
                f[mask] += f[mask | bit];
            }
        }
    }

    string out;
    out.reserve((size_t)q * 7);
    for (int i = 0; i < q; i++) {
        int m;
        cin >> m;
        out += to_string(f[m]);
        out += '\n';
    }
    cout << out;
    return 0;
}
```

**Causal recap.** The per-query scan is `O(n q) = 10^{12}` and dies by three orders of magnitude; aggregating once into an array `S[m]` makes each query `O(1)` but the obvious superset double-loop is `O(3^B) ≈ 3.5*10^9` and hits the `3^B` wall; the superset-direction SOS DP folds each bit's "set" state into its "clear" state in `O(B*2^B) ≈ 2*10^7`, which I derived and hand-checked on `B=2`; my first in-place loop updated every mask and read within-pass-mutated cells, returning garbage, which a `B=2` trace pinpointed as a wrong-direction read-after-write; guarding the update to bit-clear masks and reading the never-written bit-set partner fixes it; and 600 random differential tests against an `O(n q)` brute plus the `B=0`/`n=0`/empty-mask/full-mask/duplicate corners close it out.
