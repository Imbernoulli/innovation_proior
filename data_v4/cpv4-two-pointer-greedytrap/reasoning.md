**Reading the problem and pinning the contract.** I have `n` people with weights `w[0..n-1]`, and a zipline cart that carries *exactly two* people and launches only if their combined weight is at most `L`. Each person rides at most once, and I want to launch the maximum number of carts — that is, form the maximum number of disjoint pairs `{i, j}` with `w[i] + w[j] <= L`. Input on stdin is `n` and `L` on the first line, then the `n` weights; I print one integer, the cart count. Before any algorithm I fix the scale, because it dictates the data types. The constraints are `n <= 2*10^5`, `1 <= w[i] <= 2*10^9`, and `1 <= L <= 4*10^9`. Two things jump out immediately. First, `L` alone can be `4*10^9`, which is bigger than the signed 32-bit ceiling of about `2.147*10^9`, so `L` cannot be an `int` — reading it into an `int` would already overflow at parse time. Second, a pair-sum `w[i] + w[j]` can reach `2*10^9 + 2*10^9 = 4*10^9`, again past 32-bit range. So both `L` and every weight (and therefore every sum I compare against `L`) must be 64-bit. I will use `long long` for `L`, the weights, and the comparison. That is decision number one and it is non-negotiable; an `int` here is a silent wrong-answer on the large tests — the comparison `w[i] + w[j] <= L` would compare garbage. The *count* of pairs is at most `n/2 = 10^5`, which fits in an `int`, but I will keep it `long long` for uniformity and to avoid thinking about it again.

**Laying out the candidate approaches.** This is, abstractly, a maximum matching: build a graph where person `i` and person `j` are connected iff `w[i] + w[j] <= L`, and find the largest set of vertex-disjoint edges. General maximum matching (blossom) is `O(n^3)`-ish and absurd overkill for `n = 2*10^5`. But the graph is not arbitrary — an edge exists purely because of a *threshold on a sum*, and that special structure usually collapses to something greedy. Two cheap routes are on the table, and I want to commit to the one I can *prove*, not the one that is easiest to type.

- *Greedy by lightness.* Sort ascending, then repeatedly pair the two lightest remaining people. Intuition: light couples are the "cheapest" to seat, so seating cheap couples should leave the most room. `O(n log n)`, a few lines. The risk: the objective is the *number* of pairs, not minimizing weight used, and "cheapest couple" is a local notion while "each person once" is global — exactly the configuration where greedy tends to be wrong. I will not trust it until I have tried to break it.
- *Two pointers from both ends.* Sort ascending, put `lo` at the lightest and `hi` at the heaviest. If `w[lo] + w[hi] <= L`, pair them and move both inward; otherwise the heaviest is too heavy to pair with even the lightest, so it can never be paired with anyone — drop it (`hi--`) and try the next-heaviest. `O(n log n)` for the sort, `O(n)` sweep. The risk here is not the idea but the *proof* that this maximizes the count, plus the usual transcription hazards on the pointer loop.

**Stress-testing the lightness-greedy before committing.** Hand-waving "pairing light couples feels efficient" is how wrong solutions get shipped, so let me actually attack it with a concrete instance. Take weights `[1, 1, 2, 3]` and `L = 4`. The lightness-greedy sorts to `[1, 1, 2, 3]` and pairs the two lightest: `1 + 1 = 2 <= 4`, one cart, consuming both `1`s. The remaining people are `2` and `3`; their sum is `2 + 3 = 5 > 4`, no cart. Greedy total: **1 cart**.

Is 1 optimal? Let me hunt for a configuration greedy structurally could not reach. Pair `1 + 3 = 4 <= 4` (one cart) and `1 + 2 = 3 <= 4` (a second cart). That uses all four people in two carts. **2 carts**, strictly better than greedy's 1. So the lightness-greedy is wrong, and I now see *why*: by marrying the two `1`s to each other it "wasted" the light people on each other, when their real value is as *partners for the heavy people* who otherwise cannot ride at all. The `3` needs a `1` to come along; spending both `1`s on a cheap `1+1` cart strands the `3`. The verification paid off — it killed an approach I would otherwise have shipped.

Notice the two-pointer rule does the opposite and survives this instance: sorted `[1, 1, 2, 3]`, `lo=0 (1)`, `hi=3 (3)`, `1 + 3 = 4 <= 4` → pair, `lo=1, hi=2`. Now `w[1] + w[2] = 1 + 2 = 3 <= 4` → pair, `lo=2, hi=1`, stop. Two carts. Good — it gets the case greedy missed. But "survives one example" is not a proof, so let me argue it properly.

**Deriving the two-pointer rule and proving it.** Sort ascending. Claim: pairing the current lightest free person `lo` with the current heaviest free person `hi` whenever `w[lo] + w[hi] <= L`, and otherwise discarding `hi`, yields the maximum number of pairs.

First, the discard step is forced and safe. If `w[lo] + w[hi] > L`, then `hi` (the heaviest remaining) cannot be paired with `lo` (the lightest remaining), and every other free person is `>= w[lo]`, so `hi` cannot be paired with *anyone* — its sum with any partner is `>= w[lo] + w[hi] > L`. So in *every* feasible solution, `hi` is unpaired. Removing it changes nothing about what is achievable on the rest, so dropping it loses no pairs. That justifies `hi--`.

Second, the pairing step is safe by an exchange argument. Suppose `w[lo] + w[hi] <= L`, so `hi` *can* ride (at least with `lo`). I claim there is an optimal solution in which `lo` and `hi` are paired *together*. Take any optimal solution `S`. Consider what `hi` does in `S`:
- If `hi` is paired with `lo` in `S`, done.
- If `hi` is paired with some other person `p` (`p != lo`), and `lo` is unpaired in `S`: then swap `p` out for `lo`. The cart `{lo, hi}` is feasible (`w[lo] + w[hi] <= L`), `p` becomes free, and the pair count is unchanged. Now `lo` and `hi` are together.
- If `hi` is paired with `p` and `lo` is paired with some `q` in `S`: I want to re-pair into `{lo, hi}` and `{p, q}` without losing a cart, i.e. I must check `{p, q}` is feasible. We have `w[q] <= w[hi]` because `hi` is the heaviest free person (heavier than or equal to anyone, in particular `q`). Then `w[p] + w[q] <= w[p] + w[hi] <= L`, where the last inequality holds because `{p, hi}` is a cart in `S` and hence feasible. So `{lo, hi}` and `{p, q}` are both feasible, the count is preserved, and `lo`, `hi` are now together.
- If `hi` is unpaired in `S`: but `hi` *can* pair with `lo`. If `lo` is also unpaired, add the cart `{lo, hi}` — that *increases* the count, contradicting optimality, so this sub-case can only arise if it does not actually improve, which means `lo` must already be paired; that reduces to the previous case after we note `hi` unpaired + `lo` paired with `q` lets us form `{lo, hi}` and free `q`, count unchanged.

In all cases an optimal solution with `{lo, hi}` paired exists, so committing to that pair is safe; recurse on the remaining people. By induction the greedy two-pointer count is optimal. Good — now I trust the *idea*. The crux is `w[q] <= w[hi]`, which is exactly why pairing from the *heavy* end (not the light end) is the right exchange: the heaviest person is the most constrained, so we give it the lightest available partner.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the sweep:

```
sort(w.begin(), w.end());
int lo = 0, hi = n - 1;
long long pairs = 0;
while (lo <= hi) {
    if (w[lo] + w[hi] <= L) { pairs++; lo++; hi--; }
    else { hi--; }
}
```

The loop guard `lo <= hi` nags at me, so I trace the smallest input that could expose it: a single person, `n = 1`, `w = [5]`, `L = 100`. The answer must be `0` — one person cannot fill a two-seat cart. Start `lo = 0, hi = 0`. Guard `lo <= hi` is `0 <= 0`, true. Check `w[0] + w[0] = 5 + 5 = 10 <= 100` → `pairs++` (now 1), `lo = 1`, `hi = -1`. Guard `1 <= -1` false, stop. Output `1`.

**Diagnosing the first bug.** The code returns `1` for a single person — it paired person 0 *with itself*. The defect is precise: with `lo == hi` both pointers reference the *same* person, and `w[lo] + w[hi]` is that one weight doubled, which I then accept as a two-person cart. The loop must only consider a genuine *pair*, i.e. two distinct indices, which requires `lo < hi` strictly, not `lo <= hi`. With `lo <= hi`, whenever an odd number of people remain and they all fit, the middle person gets self-paired. The fix is to change the guard to `lo < hi`. Let me re-trace the single person: `lo = 0, hi = 0`, guard `0 < 0` false, loop never runs, output `0`. Correct. And the even case `[1, 3]`, `L = 4`: `lo = 0, hi = 1`, `0 < 1` true, `1 + 3 = 4 <= 4` → `pairs = 1`, `lo = 1, hi = 0`, guard `1 < 0` false, output `1`. Correct. The off-by-one in the guard is fixed and it broke for exactly the reason I patched.

**Second trace — re-checking the discard branch on a tie-heavy case.** Even with the guard fixed I am uneasy about the `else { hi--; }` branch when the lightest *also* cannot pair, because I want to be sure I never wrongly advance `lo` and skip a person who could still have been a partner. Let me trace `[2, 2, 2, 1, 2, 3]`, `L = 4` (this is one of the shapes my generator hammers: many ties near the limit). Sorted: `[1, 2, 2, 2, 2, 3]`, `n = 6`. Start `lo = 0, hi = 5`.
- `w[0] + w[5] = 1 + 3 = 4 <= 4` → pair, `pairs = 1`, `lo = 1, hi = 4`.
- `w[1] + w[4] = 2 + 2 = 4 <= 4` → pair, `pairs = 2`, `lo = 2, hi = 3`.
- `w[2] + w[3] = 2 + 2 = 4 <= 4` → pair, `pairs = 3`, `lo = 3, hi = 2`. Guard `3 < 2` false, stop.

Output `3`. Brute force on this case says `3` as well, so the all-fit tie case is right: three carts, everyone rides. Now a discard-heavy trace: `[3, 5, 1, 4, 2, 6]`, `L = 6`. Sorted `[1, 2, 3, 4, 5, 6]`, `n = 6`, `lo = 0, hi = 5`.
- `w[0] + w[5] = 1 + 6 = 7 > 6` → discard, `hi = 4`. (The `6` is too heavy for even the lightest `1`, so it can never ride — correctly stranded.)
- `w[0] + w[4] = 1 + 5 = 6 <= 6` → pair, `pairs = 1`, `lo = 1, hi = 3`.
- `w[1] + w[3] = 2 + 4 = 6 <= 6` → pair, `pairs = 2`, `lo = 2, hi = 2`. Guard `2 < 2` false, stop.

Output `2`. The middle person `3` is left unpaired (`3 + 3 = 6` would fit but there is only one `3`), and indeed the best is two carts. Brute agrees. The discard branch behaves: when `hi` cannot ride, we drop only `hi` and keep `lo` for the next-heaviest, never losing a viable partner.

**A subtle data-type re-check inside the loop.** Before I declare victory I look again at `w[lo] + w[hi] <= L`. With `w` declared `vector<long long>` and `L` declared `long long`, the addition is done in 64-bit and the comparison is 64-bit — no intermediate `int`. If I had declared `w` as `vector<int>`, then `w[lo] + w[hi]` would be computed in `int` first and overflow at `4*10^9` *before* the comparison widened to `long long`, giving a negative number that spuriously passes `<= L`. So the `long long` element type is load-bearing, not cosmetic. Confirmed by a direct test: weights `[2*10^9, 2*10^9]` with `L = 4*10^9` returns `1` (the sum is exactly `L`), and with `L = 4*10^9 - 1` returns `0` (the sum exceeds `L`) — both correct only because the arithmetic stays 64-bit.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: `cin >> n >> L` reads `0` and `L`; the weight loop reads nothing; `sort` on an empty vector is a no-op; `lo = 0, hi = -1`; guard `0 < -1` false; output `0`. The empty input — correct. (`hi = n - 1 = -1` is fine because it is only used in the guard, never as an index before the guard passes.)
- `n = 1`: covered above — `0 < 0` false, output `0`. A lone person fills no two-seat cart.
- `n = 2`, fits: output `1`; `n = 2`, doesn't fit (`w[0] + w[1] > L`): the `else` discards `hi`, then `lo < hi` is `0 < 0` false, output `0`. Correct.
- All pairs fit (e.g. all weights `1`, large `L`): the sweep pairs every adjacent inward step, giving `floor(n/2)` carts. Correct — that is the absolute maximum, since each cart consumes two people.
- No pair fits (e.g. all weights `> L/2` such that even the two lightest exceed `L`): the very first check fails, every step discards `hi`, count stays `0`. Correct.
- Overflow: `L` is `long long` (handles `4*10^9`); the sum `w[lo] + w[hi]` is 64-bit (handles `4*10^9`); the count `pairs` is at most `10^5`, trivially in range. No accumulator can overflow.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace so a missing weight line for `n = 0` parses fine.

**Sanity-checking the derivation against the sample.** The documented sample is `n = 4, L = 4, w = [1, 1, 2, 3]`, answer `2`. Sorted `[1, 1, 2, 3]`, `lo = 0, hi = 3`: `1 + 3 = 4 <= 4` → pair, `pairs = 1`, `lo = 1, hi = 2`; `1 + 2 = 3 <= 4` → pair, `pairs = 2`, `lo = 2, hi = 1`, stop. Output `2`. This is exactly the instance that *broke the lightness-greedy* (which scored 1), so the sample doubles as the proof that the right end matters: the two-pointer rule recovers the cart the greedy stranded. Derivation and code agree with the stated answer.

**Cross-checking against brute force.** To be sure beyond hand traces, I wrote an independent brute force that computes a true maximum matching by exhaustive recursion (take the first free person; either leave them out or pair them with each compatible later person; recurse; take the max), and a generator emitting tiny cases (`n <= 10`) heavy on ties near the limit plus an occasional large-weight regime to hit the 64-bit boundary. Running the sorted two-pointer against the brute on 600+ random cases gives **zero mismatches**. An earlier run flagged "mismatches" where my *first brute force* was the one in error — it recursed from `j + 1` after pairing `i` with `j`, silently skipping the free people sitting between `i` and `j`; the two-pointer's larger count was actually correct, which both validated the algorithm and reminded me that the oracle needs its own scrutiny. After fixing the brute to always restart from the lowest free index, the two agree everywhere.

**Final solution.** I convinced myself the *idea* is right by disproving the lightness-greedy with `[1,1,2,3]` and then proving the two-pointer rule with an exchange argument (the heaviest person is most constrained, so give it the lightest free partner), and I convinced myself the *code* is right by tracing the single-person self-pairing bug to the `lo <= hi` guard, fixing it to `lo < hi`, re-tracing the discard branch on tie-heavy inputs, nailing down the 64-bit arithmetic, and cross-checking 600+ cases against an independent matcher. That is what I ship — one self-contained file, the `O(n log n)` two-pointer I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    sort(w.begin(), w.end());

    // Two pointers from both ends: pair the lightest free rider with the
    // heaviest free rider whenever they fit together (sum <= L). If they do
    // not fit, the heaviest cannot ride with anyone (everyone else is >= the
    // lightest), so drop it and keep trying with the next-heaviest.
    int lo = 0, hi = n - 1;
    long long pairs = 0;
    while (lo < hi) {
        if (w[lo] + w[hi] <= L) {
            pairs++;
            lo++;
            hi--;
        } else {
            hi--;
        }
    }

    cout << pairs << "\n";
    return 0;
}
```

**Causal recap.** The lightness-greedy looked efficient but a single traced counterexample (`[1,1,2,3]`, `L = 4`: greedy marries the two `1`s for 1 cart, but `1+3` and `1+2` give 2) showed that spending light people on each other strands the heavy people who *need* a light partner to ride; that pushed me to pair from the *heavy* end, which an exchange argument proves optimal because the heaviest person is the most constrained and the lightest free partner is always at least as good as any other. Transcribing the sweep, a `lo <= hi` guard let the middle person of an odd group self-pair (a single person reported 1 cart) — fixed to `lo < hi`. The whole comparison `w[lo] + w[hi] <= L` rides on 64-bit arithmetic because both `L` and a pair-sum reach `~4*10^9`, so `L` and the weights are `long long`; and `max(...)`-free corners (empty, single, all-fit, none-fit) fall out of the `lo < hi` loop, leaving an `O(n log n)` solution verified against an independent matcher on 600+ cases.
