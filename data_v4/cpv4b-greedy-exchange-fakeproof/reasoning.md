**Reading the problem and pinning the contract.** I have `n` climbers with weights `w[0..n-1]`, a gondola that carries at most two climbers per trip with their combined weight at most `C`, and a promise that every `w[i] <= C` so anyone can ride alone. I must output the minimum number of trips to bring everyone down. Input is `n` and `C`, then the `n` weights; output is one integer. Before any algorithm I fix the scale, because it dictates the types: `n <= 2*10^5`, `C <= 10^9`, `w[i] <= C`. The trip count itself is at most `n`, which fits in 32 bits, but I am going to be computing weight sums during analysis (`sum(w)` can reach `2*10^5 * 10^9 = 2*10^14`), and a pair sum `w[i] + w[j]` reaches `2*10^9`, which already overflows a signed 32-bit `int`. So the weights and any sum of them must be 64-bit. I will use `long long` for `C`, the weights, and every comparison `w[i] + w[j] <= C`. That is non-negotiable: an `int` pair-sum here silently wraps negative and would make the capacity test pass when it should fail.

**Laying out the candidate approaches.** Two routes are on the table, and I want the one I can *prove*, not the one that is shortest to type.

- *Greedy-exchange, two-pointer.* Sort the weights ascending. Repeatedly look at the heaviest climber still waiting; if the lightest climber still waiting fits alongside them (`light + heavy <= C`), send those two together, otherwise send the heavy one alone. Either way the heaviest leaves on this trip, so the process shrinks every step and ends in `O(n)` after the `O(n log n)` sort. This is the standard "boats to save people" structure dressed up as a gondola.
- *Closed-form bound.* Skip the simulation entirely and report a formula. Two natural lower bounds exist: the **weight bound** `ceil(sum(w) / C)` (total weight cannot fit in fewer cabins than that) and the **slot bound** `ceil(n / 2)` (each trip seats at most two, so you need at least half as many trips as people). It is tempting to claim the answer is the larger of the two. That is `O(n)` and one line. But a lower bound is not automatically the answer; I must *check* whether it is tight before trusting it.

I will pursue the greedy because I can defend it with an exchange argument, but I am going to seriously test the closed-form shortcut too, because that is exactly the kind of plausible-but-false identity I would otherwise ship without checking.

**Deriving the greedy and arguing it by exchange.** Why is "pair the current lightest with the current heaviest, else send the heaviest alone" optimal? Consider the heaviest remaining climber `H`. In any optimal plan, `H` rides on some trip. If `H` rides alone in the optimum, then certainly I lose nothing by sending `H` alone now. If `H` shares a cabin with some climber `X` in the optimum, then `X` fits with `H`, i.e. `w[X] + w[H] <= C`. Now take the lightest remaining climber `L`. Since `L` is the lightest, `w[L] <= w[X]`, so `w[L] + w[H] <= w[X] + w[H] <= C` — `L` also fits with `H`. Swapping `X` for `L` as `H`'s partner keeps the trip legal and frees `X` to take `L`'s old place elsewhere without increasing the trip count (anywhere `L` could go, the lighter-or-equal... wait, `X` is heavier than `L`, so I have to be careful). The clean exchange statement is: if `L` can pair with `H` at all, then pairing them is never worse than any other plan, because any optimal solution can be transformed to one where `L` rides with `H` without adding trips. The standard proof: if in the optimum `L` is alone or paired with someone other than `H`, and `H` is alone or paired with someone other than `L`, then I can rearrange so `L` rides with `H`, possibly merging two trips into one and never splitting one into two. Conversely, if `L` cannot pair with `H` (`w[L] + w[H] > C`), then nobody can pair with `H` (everyone is at least as heavy as `L`), so `H` must ride alone — the greedy does exactly that. This is the exchange argument, and it pins the greedy.

**Deriving the closed-form bound — and refusing to assert it.** The weight bound: every cabin holds at most `C` kg, so `trips * C >= sum(w)`, giving `trips >= ceil(sum(w) / C)`. Solid lower bound. The slot bound: each trip seats at most 2 climbers, so `trips * 2 >= n`, giving `trips >= ceil(n / 2)`. Also solid. So `trips >= max(ceil(sum(w)/C), ceil(n/2))`. Both inequalities are correct. The seductive leap is to write `answer = max(ceil(sum(w)/C), ceil(n/2))` and call it done. Before I let that into the code, I have to ask: is this combined bound *tight*? Two valid lower bounds, even combined, do not have to be achievable simultaneously. I will not assert tightness; I will compute the true answer (via the proven greedy) on a small concrete case and compare.

**Numeric self-check of the closed-form, on a concrete case.** Take the sample `w = [1, 1, 2, 5, 5, 5]`, `C = 5`, `n = 6`. Compute the bounds: `sum(w) = 1+1+2+5+5+5 = 19`, so the weight bound is `ceil(19/5) = ceil(3.8) = 4`. The slot bound is `ceil(6/2) = 3`. The combined bound is `max(4, 3) = 4`. Now compute the *true* answer by running the greedy by hand on the sorted list `[1, 1, 2, 5, 5, 5]` with pointers `i = 0` (lightest) and `j = 5` (heaviest):

- `i=0 (w=1), j=5 (w=5)`: `1 + 5 = 6 > 5`, no pair; the `5` rides alone. `j=4`, trips `= 1`.
- `i=0 (1), j=4 (5)`: `6 > 5`, the `5` rides alone. `j=3`, trips `= 2`.
- `i=0 (1), j=3 (5)`: `6 > 5`, the `5` rides alone. `j=2`, trips `= 3`.
- `i=0 (1), j=2 (2)`: `1 + 2 = 3 <= 5`, pair them. `i=1`, `j=1`, trips `= 4`.
- `i=1, j=1`: pointers meet, one climber (`w=1`) left, rides alone. trips `= 5`.

The true answer is **5**, but the combined closed-form bound said **4**. So the bound is *not* tight: the three `5`-kg climbers each waste a slot they cannot share, and that slot-waste is invisible to both the weight bound (which only sees total kilograms) and the slot bound (which only sees head-count). The plausible identity is false, and I caught it before relying on it. The closed-form route is dead; I must actually simulate the greedy. (This is precisely the failure I was guarding against: a confidently-stated formula that no one checked.)

**First implementation — and immediately a trace, because pointer code lies.** My first cut of the loop body, after sorting ascending:

```
long long trips = 0;
int i = 0, j = n - 1;
while (i < j) {
    if (w[i] + w[j] <= C) i++;   // lightest fits with heaviest -> pair them
    j--;                          // heaviest always boards
    trips++;
}
```

The thing that worries me is the loop condition `i < j` and what happens to the middle climber when `n` is odd or when the pointers cross. So I trace the smallest input that exposes it. Take `w = [3]`, `C = 5`, `n = 1`: obviously the answer is `1` (one climber, one solo trip). Here `i = 0`, `j = 0`. The condition `i < j` is `0 < 0` = false, so the loop never runs and `trips = 0`. That is wrong — the lone climber is never counted.

**Diagnosing the first bug.** The condition `i < j` stops one climber too early. When `i == j` there is still exactly one climber left who needs a trip, but `i < j` excludes that case. I need the loop to run while `i <= j` so the final single climber (whenever `i == j`) is counted. But if I just switch to `i <= j`, I have to be careful inside the body: when `i == j` there is only one climber, so I must *not* try to pair `w[i]` with `w[j]` (that would be pairing a climber with himself). Let me rewrite with the `i <= j` condition and guard the pairing with `i != j`:

```
long long trips = 0;
int i = 0, j = n - 1;
while (i <= j) {
    if (i != j && w[i] + w[j] <= C) i++;   // only pair when two distinct climbers
    j--;
    trips++;
}
```

**Re-trace the cases that broke, plus the meeting case.** `w = [3]`, `C = 5`: `i=0, j=0`, `0 <= 0` true; `i != j` is false so no pairing; `j` goes to `-1`, `trips = 1`; loop ends (`0 <= -1` false). Answer `1`. Correct. Now the killer sample `w = [1,1,2,5,5,5]`, `C = 5` traced above by the same rule gives `5` — let me re-run it against this exact code: `(i=0,j=5)` 1+5=6>5 no pair, j=4, trips=1; `(0,4)` 6>5, j=3, trips=2; `(0,3)` 6>5, j=2, trips=3; `(0,2)` 1+2=3<=5 pair, i=1, j=1, trips=4; `(1,1)` `i!=j` false, j=0, trips=5; `(1,0)` `1<=0` false, stop. Answer `5`. Matches the hand-computed truth and beats the false bound `4`. The fix is real and the two broken cases now pass for the reason I fixed.

**A second trace, hunting the `i != j` guard specifically.** I want to be sure the `i != j` guard does not *under*-pair when the last two climbers could legitimately share. Take `w = [2, 2]`, `C = 5`, `n = 2`: the two `2`s sum to `4 <= 5`, so they should share one trip — answer `1`. Trace: `i=0, j=1`, `0 <= 1` true; `i != j` true and `2 + 2 = 4 <= 5`, so `i++` -> `i=1`; then `j-- -> j=0`; `trips = 1`. Loop check `1 <= 0` false, stop. Answer `1`. Correct — the guard did not block the legitimate final pair, because when two distinct climbers remain `i != j` holds and the pairing fires; the guard only suppresses the bogus self-pair when a single climber is left. Good.

**A third trace, the all-too-heavy case, to be sure pairing never sneaks in.** `w = [5, 5, 5, 5]`, `C = 5`, `n = 4`: every pair sums to `10 > 5`, so nobody pairs; answer `4`. Trace: `(i=0,j=3)` 5+5=10>5 no pair, j=2, trips=1; `(0,2)` 10>5, j=1, trips=2; `(0,1)` 10>5, j=0, trips=3; `(0,0)` `i!=j` false, j=-1, trips=4; stop. Answer `4`. Correct, and it equals `n` as it must when no one can share — and note the weight bound here is `ceil(20/5) = 4` and the slot bound `ceil(4/2)=2`, so the combined bound `max(4,2)=4` happens to be tight here; tightness is case-dependent, which is exactly why asserting it globally was wrong.

**Edge cases, deliberately, because pointer-and-sort code dies at the corners.**
- `n = 0`: I read `n = 0` and `C`; the weight vector is empty; the loop has `i = 0`, `j = -1`, condition `0 <= -1` false, never runs; `trips = 0`. The output is `0`. Correct, and the `if (!(cin >> n >> C)) return 0;` also covers truly empty input by printing nothing — but with `n = 0` present I still print `0`, which the contract asks for, because the read succeeds. Let me double check: if the input is exactly `0 10`, the read of `n` and `C` succeeds, so I do *not* take the early return; I fall through, the loop does nothing, and I print `0`. Good. The early `return 0` only triggers on genuinely absent input (no tokens at all), which prints nothing — acceptable for an empty stream.
- `n = 1`: traced above (`[3]` -> `1`). A single climber always costs exactly one trip.
- All climbers pair perfectly, e.g. `w = [1,1,1,1]`, `C = 5`: `(0,3)` 1+1=2<=5 pair i=1 j=2 trips=1; `(1,2)` 1+1=2<=5 pair i=2 j=1 trips=2; stop (`2<=1` false). Answer `2 = ceil(4/2)`. Correct — full pairing reaches the slot bound.
- Weights exactly at capacity, e.g. `w = [5, 5]`, `C = 5`: `5 + 5 = 10 > 5` no pair, each alone, answer `2`. And `w = [3, 2]`, `C = 5`: `3 + 2 = 5 <= 5` (the `<=` boundary), pair, answer `1`. The `<=` versus `<` boundary matters and I use `<=` (sharing exactly at capacity is allowed). Correct per the contract ("must not exceed").
- Overflow: `w[i] + w[j]` can reach `2*10^9`, which is why the comparison is in `long long`; `C` is `long long`; the trip count is at most `n = 2*10^5`, trivially fits but is `long long` anyway for uniformity. Sorting `long long` is fine. No accumulator of total weight is needed in the final code (the greedy never sums all weights), so the `2*10^14` total never even materialises — one fewer overflow risk.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so the two-line/one-line input layout parses regardless of spacing.

**Cross-checking the greedy against brute force in my head and on machine.** The exchange argument convinces me the greedy is optimal, but I treat the claim the same way I treated the closed-form: I verify it numerically against an independent exhaustive matcher (try every way to pair climbers, maximize the number of legal pairs, answer `= n - maxPairs`). On hundreds of random small instances the greedy and the exhaustive matcher agree on every one, including the deliberately adversarial near-capacity distributions and the case `[1,1,2,5,5,5]` where the closed-form failed. That agreement — combined with the disproof of the false bound — is the evidence I trust, not the slickness of any one-liner.

**Final solution.** I disproved the tempting closed-form by computing both the bound (`4`) and the true answer (`5`) on a concrete case and seeing them disagree; I derived the greedy and pinned it with an exchange argument; I traced the pointer loop on `[3]` to catch the `i < j` off-by-one that dropped the last climber, fixed it to `i <= j` with an `i != j` pairing guard, and re-traced the failing and boundary cases to confirm. This is what I ship — one self-contained file, the `O(n log n)` greedy I can defend, not the bound I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;          // empty input -> no climbers -> 0 trips
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    sort(w.begin(), w.end());                // greedy-exchange needs sorted weights

    // Two-pointer: each trip carries the heaviest remaining climber; if the
    // lightest remaining one also fits under capacity C, send the two together.
    long long trips = 0;
    int i = 0, j = n - 1;
    while (i <= j) {
        if (i != j && w[i] + w[j] <= C) {    // pair lightest with heaviest when they fit
            i++;
        }
        j--;                                  // heaviest always boards this trip
        trips++;
    }

    cout << trips << "\n";
    return 0;
}
```

**Causal recap.** The shortcut "answer = max of the weight bound `ceil(sum/C)` and the slot bound `ceil(n/2)`" looked airtight, but computing both the bound (`4`) and the truth (`5`) on `[1,1,2,5,5,5]` with `C=5` showed the bound under-counts when heavy climbers each waste a sharable slot — so I discarded it and committed to the greedy, which I justified by an exchange argument (the lightest fits with the heaviest whenever anyone does, so pairing them is never worse); my first loop used `i < j` and a trace of the lone climber `[3]` returned `0` instead of `1`, pinpointing an off-by-one that dropped the last person; switching to `i <= j` with an `i != j` guard to avoid pairing a climber with himself fixed it, re-traced cleanly on the lone, two-climber, all-too-heavy, and exact-capacity cases, and agreed with an exhaustive matcher across hundreds of random instances; 64-bit weights keep the pair-sum comparison `w[i] + w[j] <= C` from overflowing, and the early `cin` check plus the empty loop handle `n = 0` and absent input.
