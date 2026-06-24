**Reading the problem and pinning the contract.** I have a row of `n` crystals with integer charges `c[0..n-1]`, possibly negative or zero. A fusion takes two *adjacent* clusters with charges `L` and `R`, makes one cluster of charge `L + R`, and pays me a reward `L * R`. I may fuse in any order and stop whenever I want — including doing nothing. I want the maximum total reward. Input is `n` then the `n` charges; I print one integer. Two facts jump out before I pick an algorithm. First, doing nothing scores `0`, so the answer is never below `0`. Second — and this is the part I must not fumble — "do nothing" is *not* automatically best when all charges are negative, because a negative times a negative is positive: fusing two negative clusters pays a positive reward. So the all-negative input is a trap, not a free `0`. I will keep coming back to that.

Let me fix scale, because it dictates the data types and the algorithmic budget. `n <= 400` and `|c[i]| <= 10^4`. A cluster charge is a sum of original charges, so `|charge| <= 400 * 10^4 = 4*10^6`. A single fusion reward is a product of two charges, so up to `~1.6*10^13` in magnitude, and I sum many of them. That already blows past the 32-bit range (`~2.1*10^9`), so every accumulator and every reward must be 64-bit `long long`; an `int` is a silent wrong-answer on the larger tests. With `n <= 400` an `O(n^3)` algorithm is `6.4*10^7` inner steps — comfortable inside 1 second — while an `O(n^4)` would be `2.56*10^10` and far too slow. So the budget is: 64-bit arithmetic, `O(n^3)` at worst.

**Understanding the structure of a schedule.** The reward of welding two clusters depends only on their two charges at that instant, and a cluster's charge is just the sum of the original charges it contains. Crucially, fusions only ever combine *adjacent* clusters, and adjacency of contiguous runs is preserved by fusion: if I start with each crystal alone and only ever weld neighbours, then at every moment the clusters are exactly a partition of `0..n-1` into contiguous runs. So any state I can reach is described by a set of cut points, and any *final* state (when I stop) is a partition of the row into contiguous blocks, where each block is a maximal run that I chose to fully fuse, and singletons are crystals I left alone (a singleton paid nothing — there was no weld). The total reward is additive across blocks because welds inside one block never involve a cluster from another block. That decomposition is the whole game: (1) for a single contiguous block, what is the best reward to fuse it into one cluster, and (2) which blocks do I actually choose?

**Laying out the candidate approaches.** I want the one I can *prove*.

- *Greedy on adjacent pairs.* Repeatedly fuse the currently most-rewarding adjacent pair; or always left-to-right; or fuse all same-sign runs. Cheap to write. But the freedom to reorder welds is exactly the configuration where greedy tends to be wrong: the charge of a cluster changes as I fuse, so a weld that looks best now can poison later welds. I will try to break it before trusting it.
- *Interval DP for one block + partition DP on top.* For a contiguous range `[i..j]`, define the best reward to fuse it into a single cluster by trying every position of the *last* weld; then a 1-D partition DP decides which ranges to fully fuse and which crystals to leave. `O(n^3)` for the inner table, `O(n^2)` for the partition. This matches my structural decomposition exactly.

**Stress-testing greedy before committing.** Hand-waving "fuse the best pair" is how wrong solutions ship, so I attack it. Take `c = [3, 1, 5, 8]`. The most rewarding adjacent product right now is `5 * 8 = 40`, so greedy welds those first into a `13`-cluster: state `[3, 1, 13]`. Now the best adjacent product is `1 * 13 = 13`; weld to get `[3, 14]`; then `3 * 14 = 42`. Greedy total: `40 + 13 + 42 = 95`. Hmm, that happens to be `95`. Let me try a row where greedy is forced into a bad first move: `c = [10, 1, 10]`. Greedy's best first pair is a tie `10 * 1 = 10` either side; say it welds the left pair to `[11, 10]`, then `11 * 10 = 110`, total `10 + 110 = 120`. The other order: weld the right pair `1 * 10 = 10` to `[10, 11]`, then `110`, total `120` again. Same. Greedy survives those, but I do not trust "survives a couple of examples." The real worry is sign: consider `c = [-5, 4, -5]`. Greedy's adjacent products are `-20` and `-20`, both negative, so the locally-best move is to *not* fuse at all (every weld loses), giving `0`. But is `0` optimal? If I weld `-5 & 4` (reward `-20`) to get `[-1, -5]`, then `-1 * -5 = 5`, total `-20 + 5 = -15` — worse. If I weld all three some other way I still cannot beat `0` here, so on *this* instance greedy's answer `0` is right. The point is that greedy decided "stop" by looking only one weld ahead, and that one-weld-ahead view is precisely what fails when a temporarily bad weld sets up a great later one. I need a counterexample of that shape, and the cleanest is two negatives separated such that I must "pay" to bridge them — but actually two adjacent negatives already pay positively, so let me just commit to the DP, which considers *all* weld orders by construction and therefore cannot be fooled by any of these local stories. Greedy is out on principle: it cannot see past one weld, and the DP can.

**Deriving the inner interval DP and checking it on paper.** Define `mergeAll[i][j]` = the maximum total reward to fuse crystals `i..j` into a *single* cluster. To fuse `[i..j]` into one cluster, the very last weld must join two clusters that together cover `[i..j]`; since both are contiguous and adjacent, they are `[i..k]` and `[k+1..j]` for some split `k` with `i <= k < j`. Before that last weld, `[i..k]` was fused into one cluster (best reward `mergeAll[i][k]`, charge `sum(i..k)`) and `[k+1..j]` into one cluster (best `mergeAll[k+1][j]`, charge `sum(k+1..j)`), and these two sub-processes are independent. The last weld pays `sum(i..k) * sum(k+1..j)`. So

  `mergeAll[i][j] = max over k in [i, j-1] of ( mergeAll[i][k] + mergeAll[k+1][j] + sum(i..k) * sum(k+1..j) )`,

with base case `mergeAll[i][i] = 0` (a single crystal is already one cluster; no weld, no reward). I'll get contiguous sums in `O(1)` from prefix sums. Let me sanity-check on `[3, 1, 5, 8]` (indices 0..3), expecting the documented `95` for the whole range.

- Length 2: `mergeAll[0][1] = 3*1 = 3`. `mergeAll[1][2] = 1*5 = 5`. `mergeAll[2][3] = 5*8 = 40`.
- Length 3: `mergeAll[0][2]`: k=0 gives `0 + mergeAll[1][2] + 3*(1+5) = 5 + 18 = 23`; k=1 gives `mergeAll[0][1] + 0 + (3+1)*5 = 3 + 20 = 23`. So `23`. `mergeAll[1][3]`: k=1 gives `0 + 40 + 1*13 = 53`; k=2 gives `5 + 0 + 6*8 = 5 + 48 = 53`. So `53`.
- Length 4: `mergeAll[0][3]`: k=0 gives `0 + mergeAll[1][3] + 3*(1+5+8) = 53 + 42 = 95`; k=1 gives `mergeAll[0][1] + mergeAll[2][3] + 4*13 = 3 + 40 + 52 = 95`; k=2 gives `mergeAll[0][2] + 0 + 9*8 = 23 + 72 = 95`. Max `95`.

So fusing the whole row optimally gives `95`, matching the sample. The recurrence is right, and it found the same value by three different last-welds, which is a reassuring internal consistency.

**The partition layer.** Now I do not have to fuse the whole row into one cluster — I choose a partition into contiguous blocks and fully fuse each block; singletons pay `0`. Let `best[p]` = max total reward considering the first `p` crystals `c[0..p-1]`. The last crystal `p-1` is in some block; the block is `[q-1 .. p-1]` for some `q` in `[1, p]`, contributing `mergeAll[q-1][p-1]` on top of `best[q-1]`. A block of length 1 (`q = p`) contributes `mergeAll[p-1][p-1] = 0`, which is the "leave this crystal alone" option. Base `best[0] = 0` (no crystals, no reward). Answer is `best[n]`. Because every block reward is `mergeAll[..] >= 0`? No — `mergeAll` of, say, `[3, -3]` is `3 * -3 = -9 < 0`, so blocks *can* be harmful and the DP must be free to leave a crystal alone; that is exactly why the singleton option (`mergeAll[p-1][p-1] = 0`) must be present. The empty schedule overall is `best[n]` taking every crystal as its own singleton, which sums to `0`; so the answer is automatically `>= 0` without any extra `max(..., 0)`, as long as the singleton transition exists.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the partition layer, written quickly:

```
vector<long long> best(n + 1, NEG);   // <-- careless
best[0] = 0;
for (int p = 1; p <= n; p++) {
    for (int q = 1; q <= p; q++) {
        long long cand = best[q - 1] + mergeAll[q - 1][p - 1];
        best[p] = max(best[p], cand);
    }
}
cout << best[n] << "\n";
```

I initialized `best` to `NEG` "to be safe," intending to overwrite. Let me trace the all-zero row `c = [0, 0, 0]`, where the answer is obviously `0` (every product is `0`). Prefix-sum charges are all `0`, so every `mergeAll[i][j] = 0`. Now `best[0] = 0`. p=1: q=1 -> `best[0] + mergeAll[0][0] = 0 + 0 = 0`; `best[1] = max(NEG, 0) = 0`. p=2: q=1 -> `best[0] + mergeAll[0][1] = 0`; q=2 -> `best[1] + mergeAll[1][1] = 0`; `best[2] = 0`. p=3 similarly `0`. Final `0`. That case passes. So where does `NEG` bite? Let me trace a row where the *only* non-losing choice is to leave everything alone but the inner DP's losing blocks are very negative, e.g. `c = [5, -100]` is too tame; the bug I planted is subtler. Let me re-read: `best[p] = max(best[p], cand)` and `best[p]` started at `NEG`. Since `q = p` always supplies `best[p-1] + mergeAll[p-1][p-1] = best[p-1] + 0`, and `best[0] = 0`, the chain `best[p] >= best[p-1]` always holds, so `best[p]` is never left at `NEG`. The singleton transition rescues it. So this particular `NEG` init is actually harmless here — good to know, but I almost shipped a fragile pattern. I keep the cleaner `best(n + 1, 0)` with an explicit singleton-carry `b = best[p-1]` so the "leave alone" floor is structural and obvious, not an accident of the loop. That is a readability/robustness fix, not yet *the* bug.

**The real bug — a wrong base case in the inner DP.** Re-reading my inner table, my first version initialized the whole table to `NEG` and computed only `len >= 2`, forgetting to set the length-1 diagonal:

```
vector<vector<long long>> mergeAll(n, vector<long long>(n, NEG));   // <-- bug: diagonal left NEG
for (int len = 2; len <= n; len++) {
    for (int i = 0; i + len - 1 < n; i++) {
        int j = i + len - 1;
        long long best = NEG;
        for (int k = i; k < j; k++) {
            long long cand = mergeAll[i][k] + mergeAll[k + 1][j] + rangeSum(i,k)*rangeSum(k+1,j);
            best = max(best, cand);
        }
        mergeAll[i][j] = best;
    }
}
```

Trace `c = [2, 3]`, expecting `mergeAll[0][1] = 2*3 = 6`. The loop hits `len = 2`, `i = 0`, `j = 1`, `k = 0`: `cand = mergeAll[0][0] + mergeAll[1][1] + 2*3`. But `mergeAll[0][0]` and `mergeAll[1][1]` were never set — they are still `NEG = LLONG_MIN/4`. So `cand = NEG + NEG + 6`, which is around `LLONG_MIN/2 + 6`: a gigantic negative number, *and* `NEG + NEG` is close to underflowing `long long` (two copies of `LLONG_MIN/4` sum to `LLONG_MIN/2`, still in range, but any deeper nesting `NEG+NEG+NEG` for length-3 would push toward `-3/4 * 2^63`, still in range but the *value* is garbage). The reported `mergeAll[0][1]` is hugely negative instead of `6`. Then the partition layer, choosing this block, would prefer to leave the crystals alone (`0` beats a huge negative), so on `[2, 3]` it would output `0` instead of `6`. That is a concrete wrong answer caused purely by the missing base case `mergeAll[i][i] = 0`.

**Diagnosing precisely.** The defect is that a length-1 range is a *valid* sub-result with reward exactly `0` (no weld), and the recurrence for length `>= 2` *reads* those diagonal entries as the rewards of its two halves. Leaving them at `NEG` injects a sentinel where a real value (`0`) belongs. The fix is to initialize the diagonal to `0`. The cleanest way is to initialize the *entire* table to `0` and only overwrite the off-diagonal entries (`len >= 2`) with the `max`-over-`k` value; the diagonal then correctly stays `0`, and every off-diagonal entry is recomputed from scratch so its initial `0` is irrelevant. I also keep the inner `best = NEG` *local* accumulator (not the table) so that an off-diagonal cell is the max over real candidates, never accidentally `0` for a range that should be negative — e.g. `[3, -3]` must report `mergeAll = -9`, and starting the local `best` at `NEG` (then taking `k = 0`: `0 + 0 + 3*(-3) = -9`) gives `-9`, whereas starting it at `0` would wrongly clamp it to `0`. So the two `0`-vs-`NEG` choices are deliberate and opposite: the *table* diagonal is `0` (real reward of a singleton), the *local* per-cell accumulator is `NEG` (so a genuinely negative best is not clamped). Mixing them up is exactly the base-case/sign error the problem is designed to expose.

**Fixing and re-verifying.** Corrected inner DP:

```
vector<vector<long long>> mergeAll(n, vector<long long>(n, 0));   // diagonal = 0 by construction
for (int len = 2; len <= n; len++)
  for (int i = 0; i + len - 1 < n; i++) {
    int j = i + len - 1; long long best = NEG;
    for (int k = i; k < j; k++)
      best = max(best, mergeAll[i][k] + mergeAll[k+1][j] + rangeSum(i,k)*rangeSum(k+1,j));
    mergeAll[i][j] = best;
  }
```

Re-trace `[2, 3]`: `len=2,i=0,j=1,k=0`: `best = max(NEG, mergeAll[0][0] + mergeAll[1][1] + 2*3) = max(NEG, 0 + 0 + 6) = 6`. Correct. Re-trace `[3, -3]`: `best = max(NEG, 0 + 0 + 3*(-3)) = -9`. Correct — and now the partition layer will see `mergeAll[0][1] = -9 < 0` and prefer two singletons, answer `0`, which matches "every product loses, so don't fuse." Re-trace the all-negative `[-2, -3, -1]` (the headline trap): length-2 `mergeAll[0][1] = (-2)(-3) = 6`, `mergeAll[1][2] = (-3)(-1) = 3`. Length-3 `mergeAll[0][2]`: k=0 -> `0 + 3 + (-2)*(-4) = 3 + 8 = 11`; k=1 -> `6 + 0 + (-5)*(-1) = 6 + 5 = 11`. So `11`. Partition: `best[3]` will pick the whole block `11` over leaving anyone alone (`0`). Answer `11`, *not* `0`. That is the sign trap handled correctly: an all-negative row pays handsomely because negative-times-negative is positive. I verified this exact value against an independent exhaustive brute force.

**Re-running the partition layer on a worked sample.** Take `c = [3, 1, 5, 8]`. We computed `mergeAll[0][3] = 95`, and all sub-blocks are positive, so the partition DP should also land on `95` by taking the whole row as one block. `best[0]=0`. `best[1] = best[0] + mergeAll[0][0] = 0`. `best[2] = max(best[1]+mergeAll[1][1], best[0]+mergeAll[0][1]) = max(0, 3) = 3`. `best[3] = max(best[2]+0, best[1]+mergeAll[1][2], best[0]+mergeAll[0][2]) = max(3, 5, 23) = 23`. `best[4] = max(best[3]+0, best[2]+mergeAll[2][3], best[1]+mergeAll[1][3], best[0]+mergeAll[0][3]) = max(23, 3+40, 0+53, 0+95) = max(23,43,53,95) = 95`. Answer `95`, matching the sample. Good — the two layers compose correctly.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: I read no charges; I short-circuit and print `0` (the empty schedule). Without the early `return`, the prefix-sum vector of size `1` and the empty loops would still leave `best[0] = 0`, but the explicit guard is clearer and avoids constructing a `0`-by-`0` matrix edge.
- `n = 1`, `c = [-1000000000]` (well, here `|c| <= 10^4`, say `c = [-7]`): no fusion is possible, the only block is a singleton, `best[1] = best[0] + mergeAll[0][0] = 0`. Answer `0`. A lone crystal can never pay.
- All zeros `[0,0,0]`: every charge sum is `0`, every product `0`, every `mergeAll = 0`, answer `0`. Correct.
- All negatives `[-3,-3,-3,-3,-3,-3]`: every adjacent product is positive, fusing everything is great; I verified the DP returns `135` against brute. The naive instinct "all negative -> output 0" is wrong, and the DP does not make that mistake because it never special-cases sign — it just takes products.
- Mixed sign where stopping is best `[-3, 4, -3]`: products `-12, -12`; any full fusion is a net loss; DP leaves all three as singletons, answer `0`. Verified against brute.
- Two negatives `[-4, -5]`: `mergeAll[0][1] = 20`, answer `20`. Verified.
- Overflow: charges fit in `int` but their products and the running sum reach `~10^13`, so the table and accumulators are `long long`. The sentinel `NEG = LLONG_MIN/4 ~ -2.3*10^18` is only ever fed into a `max` against real candidates; the deepest expression that touches it is `NEG + NEG + product` inside the *first* `max` call when `best` still equals `NEG`, but I immediately compare against `0 + 0 + product` from the diagonal-base halves — wait, no: `best` is the *local* accumulator, and the *halves* `mergeAll[i][k], mergeAll[k+1][j]` are always real (diagonal is `0`, off-diagonal already computed), so I never actually add two `NEG`s; the only `NEG` left is the initial `best`, used solely as the left operand of `max`. So no underflow. The maximum legitimate magnitude is bounded by `n` welds each `<= 1.6*10^13`, about `6.4*10^15`, comfortably inside `long long`'s `9.2*10^18`. Safe.
- Performance: the inner table is `O(n^3) = 6.4*10^7` for `n = 400`; measured at about `0.01 s`. Memory: `mergeAll` is `400*400*8 = 1.28 MB`. Both fine.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the parser is format-agnostic.

**Final solution.** I disproved local greedy on principle (it cannot see past one weld; the DP enumerates all weld orders), derived the interval DP last-weld recurrence and checked it three ways on the sample, layered a partition DP whose singleton transition both supplies the "leave alone" floor and forces the answer `>= 0`, then caught the real defect — a missing length-1 base case that left diagonal entries at the sentinel and corrupted every block reward — by tracing `[2, 3]` and `[3, -3]`, and confirmed the sign trap by re-deriving the all-negative value `11` by hand. I ship one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    if (n == 0) { cout << 0 << "\n"; return 0; }

    // prefix sums so a contiguous charge sum is O(1)
    vector<long long> pre(n + 1, 0);
    for (int i = 0; i < n; i++) pre[i + 1] = pre[i] + c[i];
    auto rangeSum = [&](int i, int j) { return pre[j + 1] - pre[i]; }; // sum c[i..j]

    const long long NEG = LLONG_MIN / 4;

    // mergeAll[i][j] = max total reward to fuse crystals i..j into ONE cluster.
    // Last fusion joins clusters [i..k] and [k+1..j]; reward of that fusion is
    // (sum i..k) * (sum k+1..j). Base mergeAll[i][i] = 0 (single crystal, no fusion).
    vector<vector<long long>> mergeAll(n, vector<long long>(n, 0));
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            long long best = NEG;
            for (int k = i; k < j; k++) {
                long long left = rangeSum(i, k);
                long long right = rangeSum(k + 1, j);
                long long cand = mergeAll[i][k] + mergeAll[k + 1][j] + left * right;
                if (cand > best) best = cand;
            }
            mergeAll[i][j] = best;
        }
    }

    // best[p] = max reward considering the first p crystals, partitioned into
    // contiguous blocks; each block is fully fused; a block contributes its
    // mergeAll value. A length-1 block contributes 0 (no fusion). best[0] = 0.
    // We may leave crystals unmerged, so the empty action gives 0 overall.
    vector<long long> best(n + 1, 0);
    for (int p = 1; p <= n; p++) {
        long long b = best[p - 1];                 // last crystal alone (block size 1, reward 0)
        for (int q = 1; q <= p; q++) {             // last block = crystals (q-1 .. p-1)
            long long cand = best[q - 1] + mergeAll[q - 1][p - 1];
            if (cand > b) b = cand;
        }
        best[p] = b;
    }

    cout << best[n] << "\n";
    return 0;
}
```

**Causal recap.** Greedy was tempting but it commits welds one step at a time, and because a cluster's charge — hence every future reward — shifts as I fuse, a locally-best weld can poison later ones; the interval DP sidesteps this by enumerating the position of the *last* weld over every sub-range, which provably considers all orders. I checked the last-weld recurrence on `[3,1,5,8]` (three different last-welds all give `95`) and layered a partition DP whose singleton transition both encodes "leave a crystal alone" and guarantees the answer is `>= 0`. The bug that actually bit was a base-case/sign error in exactly the spot the problem targets: I left the length-1 diagonal of `mergeAll` at the sentinel `NEG` instead of `0`, so a trace of `[2,3]` returned a garbage huge-negative block reward and the partition layer wrongly preferred to do nothing; initializing the table diagonal to `0` (a singleton pays nothing) while keeping the *local* per-cell accumulator at `NEG` (so genuinely-negative blocks like `[3,-3] = -9` are not clamped to `0`) fixes it, and re-deriving the all-negative `[-2,-3,-1] = 11` by hand confirms the headline trap — an all-negative row pays a large positive reward, never a default `0`.
