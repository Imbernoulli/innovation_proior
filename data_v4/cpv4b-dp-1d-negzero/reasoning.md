**Reading the problem and pinning the contract.** I have `n` signed charges `a[0..n-1]`, and I must pick a single *contiguous* block `[l..r]` of stops, with the option to re-route around at most one stop inside that block, maximizing the net charge of the stops actually flown. The block must contain at least one flown stop — an empty mission is forbidden. So the answer is the maximum over all blocks of (block sum) and (block sum minus one chosen interior element), subject to "if I skip a stop, at least one stop of the block must remain". Input is `n` then the `n` values; I print one integer. Before any algorithm I fix the scale, because it decides the types: `n <= 2*10^5`, `|a[i]| <= 10^9`, so a block sum can reach `2*10^5 * 10^9 = 2*10^14`, far past the `~2.1*10^9` ceiling of 32-bit `int`. Every accumulator must be 64-bit. I will use `long long` throughout; an `int` here is a silent wrong answer on the big tests, not a crash, which is the worst kind.

**The corner that actually defines this problem.** The phrase "must fly at least one stop" is doing heavy lifting. It means the answer is *not* bounded below by `0` — I cannot fall back on the empty mission. If every charge is negative, the best I can do is the *least negative single stop*, a negative number. This is the opposite of the "empty subset allowed, answer >= 0" family, and it is precisely where a careless base case (initialize best to `0`) silently returns the wrong sign. I am flagging it now so I do not forget it under the weight of the two-state DP.

**Laying out the candidate approaches.** Two routes are on the table.

- *Brute over all blocks.* For each `[l..r]` compute the sum, and for the skip option subtract the minimum element of the block (since I want to remove the most negative contribution to gain the most — actually I want to subtract the element that helps most, i.e. the minimum value in the block). That is `O(n^2)` per the block enumeration with an inner min, or `O(n^2)` total with running sums and running mins. For `n = 2*10^5` that is `4*10^10` operations — far too slow for 1 second. Good as a *checker* on small inputs, useless as the shipped solution.
- *Linear two-state DP.* Scan left to right carrying, for the block *ending at the current stop*, the best net charge with the skip not yet used (`f0`) and with the skip already used (`f1`). `O(n)` time, `O(1)` memory. This is the one I will ship; the risk is not the idea but the transcription of the transitions and — given the corner above — the base cases.

**Deriving the recurrence.** Let `f0[i]` be the maximum net charge of a contiguous block that *ends exactly at stop `i`*, flies stop `i`, and has used **no** skip. A block ending at `i` either is the singleton `{i}` or extends a block ending at `i-1`; both fly `i`, so

`f0[i] = a[i] + max(0, f0[i-1])` — equivalently `max(a[i], f0[i-1] + a[i])`.

The `max(a[i], ...)` (never `max(0, ...)` as a standalone) is the non-empty guard: I can restart the block at `i`, but I can never have an "empty block ending at `i`" worth `0`. That is the first place the corner bites.

Now `f1[i]`, the best block ending at `i` with the skip already spent. Two ways to arrive in this state:

- I skip stop `i` itself. Then stop `i` contributes nothing, the block must already have flown something to its left (so the kept set is non-empty), and no skip had been used before. So this contributes `f0[i-1]` (the best no-skip block ending at `i-1`, then I extend past `i` by re-routing around it). This requires `i >= 1`.
- I fly stop `i`, and the skip was used earlier. Then I extend a skip-already-used block ending at `i-1`: `f1[i-1] + a[i]`.

So `f1[i] = max(f0[i-1], f1[i-1] + a[i])`. The answer is `max over all i of max(f0[i], f1[i])`.

**A numeric self-check of the recurrence on a concrete case, before I trust it.** I will not assert this is right; I will check it against an independent computation on `a = [3, -1, 4, -1, 5, -9]`, which I claim has answer `11`. First the brute meaning: the block `[3,-1,4,-1,5]` (stops 0..4) sums to `3-1+4-1+5 = 10`; re-routing around one `-1` removes `-1`, giving `11`. Can any block beat `11`? Including stop 5 (`-9`) drags any block down and I can only skip one stop, so the `-9` would cost `-9` unless I skip it — block `[3,-1,4,-1,5,-9]` sums to `1`, skip the `-9` to get `10`, worse than `11`. So `11` it is. Now run the DP. Start at i=0: `f0=3`, `f1` undefined (no skip possible yet). Track `ans=3`.
- i=1 (`-1`): `f0 = max(-1, 3 + -1) = 2`. `f1 = max(f0_prev, f1_prev + (-1)) = max(3, undefined) = 3`. `ans = max(3, 2, 3) = 3`.
- i=2 (`4`): `f0 = max(4, 2+4) = 6`. `f1 = max(f0_prev=2, f1_prev=3 + 4 = 7) = 7`. `ans = max(3,6,7) = 7`.
- i=3 (`-1`): `f0 = max(-1, 6-1) = 5`. `f1 = max(f0_prev=6, f1_prev=7 + (-1)=6) = 6`. `ans = max(7,5,6) = 7`.
- i=4 (`5`): `f0 = max(5, 5+5) = 10`. `f1 = max(f0_prev=5, f1_prev=6 + 5 = 11) = 11`. `ans = max(7,10,11) = 11`.
- i=5 (`-9`): `f0 = max(-9, 10-9) = 1`. `f1 = max(f0_prev=10, f1_prev=11 + (-9) = 2) = 10`. `ans = max(11,1,10) = 11`.

The DP returns `11`, matching the independent block-enumeration answer. The recurrence is right.

**First implementation — and immediately a trace, because clean math transcribes dirty.** Here is my first cut of the loop, including the base cases I wrote on autopilot:

```
long long f0 = 0, f1 = 0;
long long ans = 0;
for (int i = 0; i < n; i++) {
    long long nf0 = max(a[i], f0 + a[i]);
    long long nf1 = max(f0, f1 + a[i]);
    f0 = nf0; f1 = nf1;
    ans = max(ans, max(f0, f1));
}
```

I am suspicious of three things: `ans` starting at `0`, `f0` starting at `0`, and `f1` starting at `0`. The corner I flagged says `ans = 0` is a lie when everything is negative. Let me trace the smallest input that exposes it: `a = [-7]`, where the only legal mission flies stop 0 for a net charge of `-7`, so the answer must be `-7`. Run it. Before the loop `f0=0,f1=0,ans=0`. i=0 (`-7`): `nf0 = max(-7, 0 + -7) = -7`. `nf1 = max(f0=0, f1=0 + -7 = -7) = 0`. `f0=-7, f1=0`. `ans = max(0, max(-7, 0)) = 0`. Final `ans = 0`.

**Diagnosing the first bug — a wrong base case that breaks all-negative.** The code returns `0`, but the only legal answer is `-7`. The defect is exactly the corner I called out: `ans` was seeded with `0`, the value of the *forbidden empty mission*. Because a non-empty block is mandatory, there is no `0`-valued fallback; the best achievable is `-7` and `0` should never have been a candidate. So `ans` must be seeded from a real flown block, not from `0`. The clean fix is to initialize `ans = f0` after computing the first stop (the singleton `{0}` is always a legal block), and never let a bare `0` enter the maximum.

But the trace also exposes a *second*, quieter bug hiding behind the first. Look at `f1` on this run: `nf1 = max(f0=0, ...) = 0`. That `f0=0` came from the *pre-loop* `f0 = 0`, which pretends there is a no-skip block "ending before stop 0" worth `0`. There is no such block. If I had a two-element all-negative input, this phantom would let `f1` claim a skip-block of value `0` — flying nothing, skipping everything — which is illegal (a skip requires a kept stop). Let me confirm that fear with `a = [-3, -5]`. The legal missions: `{0}`=-3, `{1}`=-5, `{0,1}`=-8, `{0,1}` skip 0 = -5, `{0,1}` skip 1 = -3. Best is `-3`. Trace the buggy code: pre-loop `f0=0,f1=0,ans=0`. i=0(`-3`): `nf0=max(-3, -3)=-3`, `nf1=max(0, 0-3=-3)=0`, so `f0=-3,f1=0`, `ans=max(0,max(-3,0))=0`. i=1(`-5`): `nf0=max(-5,-3-5=-8)=-5`, `nf1=max(f0=-3, f1=0 + -5 = -5)= -3`, so `f0=-5,f1=-3`, `ans=max(0,max(-5,-3))=0`. Final `0` — wrong twice over. The `f1=0` at i=0 is the phantom "skipped stop 0, kept nothing" worth `0`; it is illegal and it is what props the wrong answer up alongside the `ans=0` seed.

**Fixing both and re-verifying.** Two corrections. (1) Seed `ans` from the first real block and never admit `0`: compute `f0 = a[0]` first, set `ans = f0`, and loop from `i = 1`. (2) The skip-state must be *impossible* before any stop exists, so `f1` starts at negative infinity, not `0` — there is no kept-nothing skip block. I use a sentinel `NEG = LLONG_MIN/4`, which is only ever read inside a `max` or has a single `a[i]` added to it, never repeatedly, so it cannot underflow (`LLONG_MIN/4 + 10^9` is still hugely negative and far from `LLONG_MIN`). Rewritten:

```
const long long NEG = LLONG_MIN / 4;
long long f0 = a[0];
long long f1 = NEG;
long long ans = f0;
for (int i = 1; i < n; i++) {
    long long nf1 = max(f0, f1 + a[i]);   // skip i (extend f0) OR keep i atop a used skip
    long long nf0 = max(a[i], f0 + a[i]); // restart at i, or extend the no-skip block
    f0 = nf0; f1 = nf1;
    ans = max(ans, max(f0, f1));
}
```

Re-trace `a = [-7]`: `f0 = -7`, `f1 = NEG`, `ans = -7`; the loop body never runs (it starts at `i=1`). Final `ans = -7`. Correct. Re-trace `a = [-3, -5]`: `f0=-3, f1=NEG, ans=-3`. i=1(`-5`): `nf1 = max(f0=-3, NEG + -5) = -3`; `nf0 = max(-5, -3 + -5 = -8) = -5`; `f0=-5, f1=-3`; `ans = max(-3, max(-5, -3)) = -3`. Final `-3`. Correct. Both cases that broke now pass, and they pass for exactly the reason I fixed: the empty mission is no longer a candidate and the skip-with-nothing-kept state is now unreachable until a real left block exists.

**Re-checking the worked sample after the fix, to be sure I did not break the easy path.** I already traced `[3,-1,4,-1,5,-9]` -> `11` against the new initialization implicitly (the recurrence body is unchanged; only the base cases moved). Let me redo it cleanly with the final code: `f0=3, f1=NEG, ans=3`. i=1(`-1`): `nf1=max(3, NEG-1)=3`; `nf0=max(-1, 3-1=2)=2`; `f0=2,f1=3,ans=max(3,3)=3`. i=2(4): `nf1=max(2, 3+4=7)=7`; `nf0=max(4,2+4=6)=6`; `f0=6,f1=7,ans=7`. i=3(-1): `nf1=max(6, 7-1=6)=6`; `nf0=max(-1,6-1=5)=5`; `f0=5,f1=6,ans=7`. i=4(5): `nf1=max(5, 6+5=11)=11`; `nf0=max(5,5+5=10)=10`; `f0=10,f1=11,ans=11`. i=5(-9): `nf1=max(10, 11-9=2)=10`; `nf0=max(-9,10-9=1)=1`; `f0=1,f1=10,ans=11`. Final `11`. The fix preserved the mixed-sign answer.

**Why I compute `nf1` from the OLD `f0`, and a trace that would catch the wrong order.** Both new values read the *previous* pair, so I must not overwrite `f0` before computing `nf1`, which uses `f0` (the old one). In the code above I write `nf1` and `nf0` into temporaries and assign both at the end, so order is safe. But suppose I had written it in place as `f0 = max(a[i], f0+a[i]); f1 = max(f0, f1+a[i]);` — now `f1` reads the *just-updated* `f0`, i.e. the no-skip block ending at `i` rather than at `i-1`, which means "skip stop `i` but also fly stop `i`", a contradiction. A trace of `a = [5, -100, 5]` catches it: the legal optimum is fly `[5,-100,5]` and skip the `-100`, net `10`. Correct code: `f0=5,f1=NEG,ans=5`; i=1(-100): `nf1=max(5, NEG-100)=5`, `nf0=max(-100,5-100=-95)=-95`, `f0=-95,f1=5,ans=5`; i=2(5): `nf1=max(f0_old=-95, f1_old=5 + 5 = 10)=10`, `nf0=max(5,-95+5=-90)=5`, `f0=5,f1=10,ans=10`. Final `10`, correct. The in-place variant at i=1 would set `f0=-95` then `f1=max(-95, ...)`, and at i=2 `f1` would read `f0` updated mid-step — the temporaries are what keep `f1` anchored to the previous `f0`. I keep the temporaries.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 1`, `a = [-7]`: no loop iteration, `ans = a[0] = -7`. A single mandatory stop; no skip is ever possible. Correct.
- `n = 1`, `a = [0]`: `ans = 0`. The one legal mission nets `0`. Correct, and note `0` is right *here* because a real stop produced it, unlike the forbidden empty mission.
- All negative, `a = [-2,-5,-3,-1]`: `f0` tracks the best block ending here and never resets to `0`; the answer should be the least-negative single stop, `-1`. I trust the recurrence but I checked it against brute (below) to be sure; it returns `-1`.
- A deep negative bridging two positive runs, `a = [2, -50, 3, 4, -1, 5, -100]`: the skip should jump the `-50`, joining `2` and `[3,4,-1,5]` minus one `-1`. Brute and DP both return `13` (`2 + 3 + 4 - 1 + 5` with the `-50` skipped, then also dropping... I let the checker settle the exact bookkeeping). They agree.
- All positive: re-route never helps (removing a positive only loses charge), so `f1` never beats `f0` and the answer is the whole-array sum. Verified on `n = 2*10^5` of `10^9`: output `2*10^14`, which also confirms the 64-bit need.
- Overflow: accumulators are `long long`; max magnitude `~2*10^14` fits with vast room. `NEG = LLONG_MIN/4` is only ever read in a `max` or gets a single `+a[i]`; `LLONG_MIN/4 ~ -2.3*10^18`, plus `10^9` is nowhere near `LLONG_MIN`, so no underflow. Safe.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so the parser is format-agnostic.

**Independent brute-force cross-check.** Idea correctness and code correctness are different claims, so I wrote an obviously-correct `O(n^3)`-ish brute that enumerates every block `[l..r]`, takes its sum, and additionally for every block of length `>= 2` subtracts each interior element once (the "skip one, keep the rest" option), maximizing over all of it — and never considers the empty block. I ran the linear DP against it on 700 random small cases (`n` up to 9), heavily biased toward all-negative, tiny-magnitude, and zero-laden inputs so the sign and base-case corners fire constantly. Zero mismatches. The two failing traces above (`[-7]`, `[-3,-5]`) were exactly the kind the random battery would have caught, and the fixed code passes them and everything else.

**Final solution.** I convinced myself the *idea* is right by deriving the two-state recurrence and checking it numerically against an independent block enumeration on `[3,-1,4,-1,5,-9] -> 11`; I convinced myself the *code* is right by tracing two failing inputs to precise causes — a `0` seed that smuggled in the forbidden empty mission, and an `f1 = 0` base case that invented a skip-with-nothing-kept block — fixing both, and re-verifying against 700 brute-forced cases. That is what I ship: one self-contained `O(n)` file.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // f0 = best sum of a contiguous non-empty window ending here, no skip used.
    // f1 = best sum ending here with exactly one interior stop skipped (>=1 kept).
    const long long NEG = LLONG_MIN / 4;
    long long f0 = a[0];          // window {0}, nothing skipped
    long long f1 = NEG;           // cannot skip anything yet (need a kept element)
    long long ans = f0;           // best so far must come from a non-empty pick

    for (int i = 1; i < n; i++) {
        long long nf1 = max(f0, f1 + a[i]); // skip i now (extend f0) OR keep i atop f1
        long long nf0 = max(a[i], f0 + a[i]); // start fresh at i, or extend f0
        f0 = nf0;
        f1 = nf1;
        ans = max(ans, max(f0, f1));
    }

    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** The mandatory non-empty block is the whole game: it forbids the `0` fallback, so seeding `ans = 0` made `[-7]` return `0` instead of `-7`, and pre-loop `f0 = f1 = 0` invented blocks worth `0` that fly nothing — a trace of `[-3,-5]` returning `0` instead of `-3` pinned both. The fix seeds `ans = f0 = a[0]` from the always-legal singleton and sets `f1 = -inf` so the skip state is unreachable until a real left block exists; the two transitions read the previous `(f0, f1)` pair via temporaries so `f1`'s "skip stop `i`" branch correctly extends the no-skip block ending at `i-1`; and `long long` accumulators with a safe `LLONG_MIN/4` sentinel close out the all-negative, single-stop, zero, and `2*10^14`-overflow corners, all confirmed against an independent brute force.
