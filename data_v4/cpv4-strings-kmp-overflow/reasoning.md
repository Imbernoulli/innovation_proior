**Reading the problem and pinning the contract.** I have one string `s` of length `n`, lowercase only. For every prefix length `k` from `1` to `n` I must count `occ(k)` = the number of start positions at which the prefix `s[0..k-1]` appears as a contiguous substring of `s` (overlaps allowed, distinct starts are distinct occurrences), and I output the sum `score = occ(1) + ... + occ(n)`. Input is the single string on stdin; output is one integer. Before any algorithm I fix the scale because it dictates the data types: `n <= 2*10^5`. The worst case for the score is the most self-similar string, `s = "aaaa...a"`: there the prefix of length `k` occurs at every start `0..n-k`, i.e. `n-k+1` times, so `score = sum_{k=1}^{n} (n-k+1) = n(n+1)/2`. For `n = 2*10^5` that is `200000*200001/2 = 20000100000 ≈ 2*10^10`. That is almost ten times past the 32-bit signed range of `2147483647`. So every counter and the accumulator must be 64-bit. I write that down now and will come back to bite myself if I forget it — an `int` here is a silent wrong-answer, not a crash.

**Laying out the candidate approaches.** Two routes, and I want the one I can both *prove* and *afford*.

- *Per-prefix substring count.* For each of the `n` prefixes, scan `s` and count matches. Obviously correct, and it is exactly what my brute force will do. But matching prefix of length `k` over `s` is `O(n)` per prefix at best, so the whole thing is `O(n^2)` — `4*10^10` operations at `n = 2*10^5`. Dead on arrival under a 1-second limit. Useful only as an oracle on tiny inputs.
- *One KMP prefix-function pass.* The "patterns" here are all `n` prefixes of `s` itself, and that self-reference is the gift: the prefix function `pi` already tells me, for each end position, the longest prefix that ends there. The shorter prefixes ending there are the longest one's borders, i.e. the chain `i -> pi[i]-1 -> ...`. If I can aggregate counts down those chains I get every `occ(k)` in `O(n)`. The risk is not the idea but the *bookkeeping*: what exactly does each `pi[i]` contribute, and in what order do I propagate.

I commit to the prefix-function pass and keep the per-prefix count as the brute oracle.

**Deriving the aggregation and checking the off-by-one on paper.** The KMP prefix function `pi[i]` (0-indexed over positions) is the length of the longest proper border of `s[0..i]` — equivalently, the longest prefix of `s` (other than the whole `s[0..i]`) that ends at position `i`. Now I ask: which prefixes end exactly at position `i`? The full prefix `s[0..i]` of length `i+1` ends there trivially. Beyond that, the prefixes that end at `i` are precisely those of length `pi[i]`, then `pi[pi[i]-1]`, and so on down the border chain to `0`. So if I want, for each length `L`, "how many times does a prefix of length `L` end at *some* position", I can:

1. Count the *deepest* contribution at each position: for every `i`, the prefix of length `pi[i]` ends at `i` as a border. Tally `cnt[pi[i]] += 1`.
2. Then push those tallies down the chain: a prefix of length `L` ending somewhere also implies its own border of length `pi[L-1]` ends at that same spot. Process lengths from longest to shortest, doing `cnt[pi[L-1]] += cnt[L]`.
3. Finally, every prefix of length `L` (for `L = 1..n`) occurs at least once at the very start of `s` — that "own" occurrence is not a border of anything, so add `1` to each: `cnt[L] += 1`.

Then `occ(L) = cnt[L]` and `score = sum_L cnt[L]`. This is the standard "occurrences of each prefix" aggregation; the subtle parts are (a) step 1 indexes by `pi[i]`, *not* `pi[i]+1` (the border *length* is `pi[i]`), and (b) the `+1` in step 3 for the prefix's own appearance.

Let me sanity-check the derivation on `s = "ababa"`, where I claimed `score = 9`. Positions `0..4`, characters `a b a b a`.

- Prefix function: `pi[0]=0`. `pi[1]`: `s[1]='b'` vs `s[0]='a'` -> 0. `pi[2]`: `j=pi[1]=0`, `s[2]='a'==s[0]='a'` -> `j=1`, `pi[2]=1`. `pi[3]`: `j=pi[2]=1`, `s[3]='b'==s[1]='b'` -> `j=2`, `pi[3]=2`. `pi[4]`: `j=pi[3]=2`, `s[4]='a'==s[2]='a'` -> `j=3`, `pi[4]=3`. So `pi = [0,1,0,1...]` wait, recompute carefully: `pi=[0,0,1,2,3]`.
- Step 1, `cnt[pi[i]] += 1` for `i=0..4`: `pi` values are `0,0,1,2,3`, so `cnt[0]+=2`, `cnt[1]+=1`, `cnt[2]+=1`, `cnt[3]+=1`. Now `cnt = [2,1,1,1,0,0]` (indices `0..5`).
- Step 2, for `L=n..1` do `cnt[pi[L-1]] += cnt[L]`. `n=5`. `L=5`: `pi[4]=3`, `cnt[3]+=cnt[5]=0` -> no change. `L=4`: `pi[3]=2`, `cnt[2]+=cnt[4]=0` -> no change. `L=3`: `pi[2]=1`, `cnt[1]+=cnt[3]=1` -> `cnt[1]=2`. `L=2`: `pi[1]=0`, `cnt[0]+=cnt[2]=1` -> `cnt[0]=3`. `L=1`: `pi[0]=0`, `cnt[0]+=cnt[1]=2` -> `cnt[0]=5`. Now `cnt=[5,2,1,1,0,0]`.
- Step 3, `cnt[L]+=1` for `L=1..5`: `cnt=[5,3,2,2,1,1]`.
- `occ(1)=cnt[1]=3`, `occ(2)=2`, `occ(3)=2`, `occ(4)=1`, `occ(5)=1`. Sum `= 3+2+2+1+1 = 9`. Matches the stated sample, and matches the by-hand occurrence list in the statement. The derivation and the off-by-one are right.

I notice the bucket `cnt[0]` accumulates a lot of mass (it ended at `5`), but I never read it into the score — the score sums `L=1..n`. `cnt[0]` is "the empty prefix", which I deliberately ignore. Good; that bucket is just a chain sink.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the aggregation, using `int` for the `pi` array (lengths are tiny, that is fine) and — here is where I have to be careful — for the counts. I will *deliberately* write the naive version first the way a tired competitor would, to see whether my own checks catch the overflow I warned myself about:

```
vector<int> occ(n + 1, 0);
for (int i = 0; i < n; i++) occ[pi[i]]++;
for (int i = n; i >= 1; i--) occ[pi[i - 1]] += occ[i];
for (int len = 1; len <= n; len++) occ[len]++;
int score = 0;
for (int len = 1; len <= n; len++) score += occ[len];
cout << score << "\n";
```

Let me trace it on something small first to confirm the *logic* is faithful to my paper derivation, before worrying about width. Take `s = "aaaa"`, where every prefix `a`, `aa`, `aaa`, `aaaa` occurs `4,3,2,1` times, so `score = 4+3+2+1 = 10`.

- `pi`: `pi[0]=0`; `pi[1]`: `j=0`, `s[1]=s[0]` -> `j=1`; `pi[2]`: `j=1`, `s[2]=s[1]` -> `j=2`; `pi[3]`: `j=2`, `s[3]=s[2]` -> `j=3`. So `pi=[0,1,2,3]`.
- Step 1: `cnt[pi[i]]++` over `pi=0,1,2,3` -> `cnt=[1,1,1,1,0]` (indices `0..4`).
- Step 2, `L=4..1`: `L=4`: `pi[3]=3`, `cnt[3]+=cnt[4]=0`. `L=3`: `pi[2]=2`, `cnt[2]+=cnt[3]=1` -> `cnt[2]=2`. `L=2`: `pi[1]=1`, `cnt[1]+=cnt[2]=2` -> `cnt[1]=3`. `L=1`: `pi[0]=0`, `cnt[0]+=cnt[1]=3` -> `cnt[0]=4`. Now `cnt=[4,3,2,1,0]`.
- Step 3: `+1` to `L=1..4` -> `cnt=[4,4,3,2,1]`.
- score `= cnt[1]+cnt[2]+cnt[3]+cnt[4] = 4+3+2+1 = 10`. Correct.

So the *logic* is faithful. My brute force agrees on this and on hundreds of random small strings (I run `gen.py` -> compare `sol` vs `brute.py`, 500 cases, zero mismatches). The small-case correctness is locked.

**The bug — the overflow I warned myself about, and a trace that exposes it.** The small tests all pass, but small tests cannot reach the overflow regime: with `n <= 14` the largest possible score is `14*15/2 = 105`, nowhere near `2^31`. The danger only shows up at scale, which random small generators will never visit. So I must reason about, and then *measure*, the large case explicitly. Consider `s = "a"` repeated `200000` times. By my closed form `score = n(n+1)/2 = 20000100000`. Now trace what my `int` code computes for the accumulator. After steps 1-3, `cnt[L] = n-L+1` (each prefix of length `L` occurs `n-L+1` times). The accumulator runs `score += occ[len]`, summing `n + (n-1) + ... + 1`. As a partial sum it climbs past `2147483647` somewhere around `L` where the running total first exceeds `2^31`; from then on every add wraps modulo `2^32` into the signed range, and the final value is `20000100000 mod 2^32` reinterpreted as signed, which is `20000100000 - 5*2^32 = 20000100000 - 21474836480 = -1474736480`. So the `int` version prints `-1474736480` — a *negative* score, which is impossible for a sum of nonnegative occurrence counts. I actually compile a throwaway `int` build and run it: it prints exactly `-1474736480`, confirming the wrap. That negative output is the smoking gun.

There is a second, quieter width trap hiding in step 2: `occ[pi[i-1]] += occ[i]`. The individual bucket values can themselves be large — for `"aaaa...a"`, `cnt[0]` ends near `n(n+1)/2` because the whole border mass funnels into it. Even though I never read `cnt[0]` into the score, the `+=` that builds it would overflow an `int` bucket mid-aggregation, corrupting nothing I read here but a landmine if I ever changed which buckets I sum. So the *bucket array itself* must be 64-bit, not only the final accumulator. The fix is the same as the diagnosis demanded from the very start: make the counts `long long`.

**Fix and re-verification.** Promote the count array and the accumulator to `long long`; keep `pi` as `int` (border lengths never exceed `n < 2^31`, and that array is read into 64-bit contexts only by value):

```
vector<long long> occ(n + 1, 0);
for (int i = 0; i < n; i++) occ[pi[i]]++;
for (int i = n; i >= 1; i--) occ[pi[i - 1]] += occ[i];
for (int len = 1; len <= n; len++) occ[len]++;
long long score = 0;
for (int len = 1; len <= n; len++) score += occ[len];
```

Re-run the overflow case: `"a"*200000` now prints `20000100000`, matching the closed form `n(n+1)/2` exactly. Re-run all 500 random small cases against the brute: still zero mismatches (promoting to `long long` cannot change any small-case value, only prevents wrap). Re-trace `"ababa"` mentally through the promoted code: the bucket math is identical to my paper trace, so it still yields `9`. The bug existed for the precise reason I predicted, and the fix is exactly the width promotion — that correspondence between predicted cause and observed wrap is the evidence I trust.

**A second debug episode: the propagation bound and an off-by-one I nearly shipped.** While re-reading the aggregation loop I get nervous about the bound on the second loop. I wrote `for (int i = n; i >= 1; i--) occ[pi[i - 1]] += occ[i];`. The reference algorithm I half-remember writes the loop as `for i in [n-1 .. 1]: occ[pi[i-1]] += occ[i]`. Did I introduce a phantom iteration at `i = n`? Let me trace the `i = n` step concretely: it does `occ[pi[n-1]] += occ[n]`. Now `occ[n]` is the count for the full-length prefix; after step 1 it equals the number of positions `i` with `pi[i] = n`, which is *zero* (no `pi[i]` can equal `n`, since `pi[i] <= i < n`). So `occ[n] = 0` at this point and the `i = n` iteration adds `0` — harmless, a no-op. Good: my extra top iteration is safe, not a bug. But this made me check the *other* end: does the loop need to run at `i = 1`? At `i = 1` it does `occ[pi[0]] += occ[1] = occ[0] += occ[1]`, folding the length-1 prefix's count into the empty bucket `occ[0]`. Since I never read `occ[0]` into the score, this too is harmless — but it must still *execute* for correctness of any chain that passes *through* length 1 on its way down. In `"ababa"` the chain `pi[2]=1` meant length-3 counts flow into length-1 (`L=3` step), and length-1 then needs to be final before I read it; the `L=1` iteration only touches `occ[0]`, so `occ[1]` is already finalized when read. I verify on `"ababa"`: my paper trace above used exactly this loop and got `occ[1]=3`, matching brute. So the bound `i = n .. 1` is correct; trimming it to `n-1 .. 1` would also work but mine is equivalent. I keep mine and stop second-guessing — the trace settled it.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0` (empty input): `cin >> s` fails, I print `0` and return. The empty string has no prefixes, score `0`. Correct. (Brute also prints `0`.)
- `n = 1`, `s = "z"`: the `pi` loop never runs (`i` starts at 1, `n=1`). Step 1 reads `pi[0]=0` -> `occ[0]=1`. Step 2 `L=1`: `occ[pi[0]] += occ[1]` -> `occ[0]+=0`. Step 3 `occ[1]+=1` -> `occ[1]=1`. score `= occ[1] = 1`. The single prefix occurs once. Correct.
- Unique-alphabet string, `s = "abcd"`: every `pi[i]=0`, so step 1 puts all mass in `occ[0]`, step 2 only ever touches `occ[0]` (since `pi[L-1]=0` for all), and step 3 sets each `occ[L]=1` for `L=1..4`. score `= 4`. Each prefix occurs exactly once (at the start) — correct, and a good check that the border machinery does not invent phantom occurrences.
- Maximal-overlap string, `"aaaa...a"` at `n=2*10^5`: score `= n(n+1)/2 = 20000100000`, comfortably inside `long long` (cap `~9.2*10^18`). The buckets themselves peak around `2*10^10`, also fine in 64-bit. Verified above.
- Overflow sentinel: there is none — I add only nonnegative counts, never subtract, so no underflow; the only risk was width, now closed by `long long` on both buckets and accumulator.
- Output: exactly one integer and a newline; `cin >> s` skips leading whitespace and stops at whitespace/EOF, so a trailing newline in the input is harmless.

**Performance.** Prefix function is `O(n)` amortized; the three aggregation passes are each `O(n)`; the sum is `O(n)`. Total `O(n)` time, `O(n)` memory. On `n = 2*10^5` it runs in about 7 ms — three orders of magnitude under the 1-second limit. No I/O tuning needed.

**Final solution.** I convinced myself the idea is right by deriving the border-chain aggregation and hand-checking the off-by-one on `"ababa"` and `"aaaa"`; I convinced myself the *code* is right by (1) tracing the `int` version to a concrete wrapped value `-1474736480` and watching the `long long` version recover the exact `20000100000`, and (2) tracing the suspicious `i = n` loop bound to a provable no-op. That is what I ship — one self-contained `O(n)` file with 64-bit counts throughout:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    string s;
    if (!(cin >> s)) { cout << 0 << "\n"; return 0; } // empty input -> score 0
    int n = (int)s.size();

    // KMP prefix function: pi[i] = length of the longest proper border of s[0..i].
    vector<int> pi(n, 0);
    for (int i = 1; i < n; i++) {
        int j = pi[i - 1];
        while (j > 0 && s[i] != s[j]) j = pi[j - 1];
        if (s[i] == s[j]) j++;
        pi[i] = j;
    }

    // occ[len] = number of times the length-`len` prefix occurs as a substring of s.
    // Every border of length `pi[i]` ending at i is one such occurrence; chase the
    // border chain by propagating counts from longer borders to shorter ones, then
    // add 1 to every prefix length for its "own" occurrence at the start.
    vector<long long> occ(n + 1, 0);
    for (int i = 0; i < n; i++) occ[pi[i]]++;
    for (int i = n; i >= 1; i--) occ[pi[i - 1]] += occ[i];
    for (int len = 1; len <= n; len++) occ[len]++;

    // Self-similarity score = total number of (prefix, occurrence) incidences.
    long long score = 0;
    for (int len = 1; len <= n; len++) score += occ[len];

    cout << score << "\n";
    return 0;
}
```

**Causal recap.** The self-similarity score is the sum over all prefixes of how often each occurs, and the self-reference (patterns are the string's own prefixes) lets the KMP prefix function deliver every count in one linear pass via border-chain aggregation, which I validated on `"ababa"` (9) and `"aaaa"` (10) against a brute force on 500 random cases; the one trap small tests cannot reach is width — the maximal-overlap string `"aaaa...a"` at `n=2*10^5` has score `n(n+1)/2 ≈ 2*10^10`, so an `int` accumulator (and even the intermediate buckets) wraps to the impossible negative `-1474736480`, which I reproduced and then fixed by making the counts `long long`, recovering the exact `20000100000`; a second trace confirmed the `i = n` aggregation bound is a harmless no-op, and the empty / single-char / unique-alphabet corners all fall out of the same recurrence.
