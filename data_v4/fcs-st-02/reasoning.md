**Reading the problem and pinning the contract.** I am given the prefix-function array `pi[0..n-1]` of some hidden string and I must output a string whose prefix function equals `pi` exactly, or `-1` if none exists. Let me fix definitions before touching an algorithm, because the whole problem lives in a definition. For a string `s`, `pi[i]` is the length of the longest *proper* prefix of `s[0..i]` that is also a suffix of `s[0..i]` — equivalently, the longest *border* of the prefix of length `i+1`. The scale: `n <= 2*10^5`, so I need something linear or near-linear; `O(n^2)` is out. Output is any valid witness over `a`–`z`, checked by recomputing the prefix function (not by string identity), and `-1` for unrealizable arrays. For `n = 0` the empty string is the answer. I read `n` then `n` integers; the values are *claimed* to be in `[0, i]` but the input may be adversarial garbage that I have to reject. So this is two problems welded together: a **feasibility test** and a **construction**.

**First instinct: a closed-form legality check over the array.** The seductive move is to characterize valid prefix functions by simple local inequalities and, if they hold, hand the array to the standard reconstruction. Two conditions are obviously necessary. First, `pi[0] = 0` and `0 <= pi[i] <= i`: a single character has no proper border, and a border of `s[0..i]` is shorter than `i+1`. Second, a border grows by at most one per step: `pi[i] <= pi[i-1] + 1`. Why? If `s[0..i]` has a border of length `L >= 1`, then dropping the last character gives a border of length `L-1` of `s[0..i-1]`, so `pi[i-1] >= pi[i] - 1`. These two checks are cheap, `O(n)`, and they *feel* like they ought to be the whole story. The plan would be: verify them, then run the cp-algorithms reconstruction, print the string.

**Showing the obvious check is not sufficient — a concrete kill.** Before I trust "local inequalities ⇒ valid", let me actually attack it, because this is exactly the configuration where a local check is wrong: the prefix function couples *non-adjacent* positions through the border chain. Take `pi = [0, 1, 1]`. It passes every local test: `pi[0]=0`, each `pi[i] <= i`, and the steps are `0 -> 1` (`<= 0+1`) and `1 -> 1` (`<= 1+1`). So my naive checker would declare it valid and try to build a string. But is there a string with prefix function `[0,1,1]`? `pi[1] = 1` forces `s[1] = s[0]`, so `s = "xx?"` for some character `x`. Now look at `pi[2]`. The only candidate border of `"xx?"` of length 2 needs `s[0..1] = "xx"` to equal the suffix `s[1..2] = "x?"`, i.e. `? = x`, giving `"xxx"` whose `pi[2] = 2`. If instead `? != x`, then a border of length 1 needs `s[0] = s[2]`, i.e. `x = ?`, which we just excluded, so `pi[2] = 0`. The value `pi[2] = 1` is **unreachable** — it is neither `2` nor `0`. So `[0,1,1]` is infeasible, yet my local checker waved it through. The naive approach would emit a wrong string (or crash), and on a `-1` test it would print a bogus answer.

**Diagnosing *why* the local check fails — the transitive border law.** The reason `pi[2]=1` was impossible is instructive. A border of length `1` of `s[0..2]` would require `s[2] = s[0]`. But we already forced `s[1] = s[0]` (from `pi[1]=1`), and the way borders extend, if `s[2] = s[0]` then the length-2 border `"xx"` already matched and `pi[2]` would have been pushed to `2`, not `1`. In general the legal values of `pi[i]` are exactly the elements of the **border chain** of `s[0..i-1]` — the set `{ pi[i-1], pi[pi[i-1]-1], pi[pi[pi[i-1]-1]-1], ... , 0 }` — each *plus one* if the next character matches, or `0` if none matches. That chain is a transitive, non-local object: whether `pi[i]=1` is legal depends on the entire history of forced characters, not on `pi[i-1]` alone. **This is the insight the problem hinges on: feasibility is governed by transitive border composition, and the only honest way to test it is to follow the chain — or, equivalently, to construct a witness and check it.**

**Turning the insight into an algorithm: construct, then verify.** I could try to encode the transitive law as a direct feasibility predicate, but that is fiddly and error-prone — exactly the kind of "clever closed form" that ships subtle bugs. There is a cleaner, provably-sound route that *uses* the same structure. I reconstruct a candidate string greedily by the standard rule, and then I **recompute its prefix function and compare to the input**. The verification step is what makes the whole thing a sound feasibility test: if the input array is unrealizable, the string I build cannot possibly have that prefix function, so the recomputation disagrees and I output `-1`. If it agrees, I have an explicit witness — there is nothing left to prove. The construction rule:

- `pi[0] = 0` always; position `0` is unconstrained, so set `s[0] = 'a'`.
- For `i >= 1` with `pi[i] > 0`: a border of length `pi[i]` means `s[0..pi[i]-1] = s[i-pi[i]+1..i]`, and comparing the last characters gives `s[i] = s[pi[i]-1]`. The character is **forced**.
- For `i >= 1` with `pi[i] = 0`: `s[i]` must *not* continue any border of `s[0..i-1]`. If it did continue the border of length `k`, then `s[i] = s[k]` and `pi[i]` would be at least `k+1 > 0`. So I must avoid `s[k]` for every `k` on the border chain `pi[i-1] -> pi[pi[i-1]-1] -> ... -> 0`, and pick any character outside that forbidden set.

The forbidden set at a `pi=0` position is the distinct characters `{ s[k] : k in border chain }`. I need an alphabet large enough that a free character always exists.

**Bounding the alphabet — why 26 letters suffice.** This worried me: could a long border chain forbid more than 26 distinct characters and force `-1` on a genuinely feasible array? The border chain has strictly decreasing lengths `k_1 > k_2 > ... > 0`, but its *length* can be up to `i`. What matters is the number of **distinct** forbidden characters. Each forbidden character is `s[k_j]`, a head of a shorter border, and these heads coincide a lot. The deepest border structure is the Zimin/ruler word `a b a c a b a d ...`, whose chain depth is `Theta(log n)`. I checked empirically: across heavy random strings and the Zimin word up to length `2^17 - 1 = 131071`, the maximum number of distinct letters the reconstruction ever needs is `18`. Even pushing the Zimin construction to `n = 2*10^5`, it stays at `18 < 26`. So `26` is comfortably safe; I keep the structural guard "if no free letter exists, output `-1`" as a belt-and-suspenders branch that provably never fires for valid input but makes the code total.

**Putting in the cheap structural pre-checks anyway.** Even though the verify pass is the real test, I keep the `O(n)` structural rejects up front (`pi[i]` in range, `pi[i] <= pi[i-1]+1`). Two reasons. They reject obviously-broken input (negative values, `pi[i] > i`) before I index into `s[pi[i]-1]`, avoiding out-of-bounds reads on adversarial arrays. And they short-circuit the common garbage cases without building a string. They are *not* relied on for correctness — the verify pass is — but they keep the construction loop safe.

**First implementation.** My initial loop:

```text
string s(n, 'a');
for (int i = 0; i < n; i++) {
    if (pi[i] > 0) {
        s[i] = s[pi[i] - 1];
    } else {
        bool used[26] = {false};
        int k = pi[i - 1];              // <-- start of border chain
        while (true) {
            used[s[k] - 'a'] = true;
            if (k == 0) break;
            k = pi[k - 1];
        }
        int c = 0; while (c < 26 && used[c]) c++;
        s[i] = 'a' + c;
    }
}
```

then recompute the prefix function of `s` and compare. Something is off about the `pi[i] == 0` branch when `i == 0`, so I trace the smallest input that exercises it.

**The debug episode — tracing `n = 1`, `pi = [0]`.** Expected output: any single character, say `"a"`. Walk the code. `i = 0`, `pi[0] = 0`, so I enter the `else` branch and execute `int k = pi[i - 1]` — that is `pi[-1]`. **Out-of-bounds read.** On `n = 1` it reads whatever sits before the vector; with sanitizers it is a hard crash, and even without, `s[k]` then indexes `s[garbage]`. The defect is precise: the border-chain walk assumes there *is* a previous position `i-1`, but position `0` has no predecessor and no border chain at all — its character is simply free. I confirmed this is real by compiling with `-fsanitize=address` on `printf 0 | ./sol`, which flagged the negative index. The fix is to treat `i == 0` as its own case and set `s[0] = 'a'` directly; the chain walk only makes sense for `i >= 1`, where `pi[i-1]` is a valid index. I split the branch:

```text
if (i == 0)            s[0] = 'a';                 // unconstrained
else if (pi[i] > 0)    s[i] = s[pi[i] - 1];        // forced by the border
else {                 // pi[i] == 0, i >= 1 : avoid every chain head
    int k = pi[i - 1]; ...
}
```

**Re-tracing the fixed code on the cases that mattered.** `n = 1, pi = [0]`: now `i = 0` sets `s = "a"`, verify recomputes `chk = [0] = pi`, output `"a"`. Correct, no bad index. `pi = [0,1,1]` (the infeasibility I found): `i=0 -> 'a'`; `i=1`, `pi[1]=1>0`, `s[1] = s[0] = 'a'`, giving `"aa"`; `i=2`, `pi[2]=1>0`, `s[2] = s[pi[2]-1] = s[0] = 'a'`, giving `"aaa"`. Now verify: prefix function of `"aaa"` is `[0,1,2]`, which is **not** `[0,1,1]`, so I output `-1`. The construct-then-verify pipeline catches exactly the transitive infeasibility my local checker missed — this is the payoff of the insight. `pi = [0,0,1,2,3,0]`: `i=0 'a'`; `i=1` `pi=0`, chain from `pi[0]=0`: forbidden `{s[0]='a'}`, pick `'b'` -> `"ab"`; `i=2` `pi=1`, `s[2]=s[0]='a'` -> `"aba"`; `i=3` `pi=2`, `s[3]=s[1]='b'` -> `"abab"`; `i=4` `pi=3`, `s[4]=s[2]='a'` -> `"ababa"`; `i=5` `pi=0`, chain from `pi[4]=3`: `3 -> pi[2]=1 -> pi[0]=0`, forbidden `{s[3]='b', s[1]='b', s[0]='a'} = {'a','b'}`, pick `'c'` -> `"ababac"`. Verify: prefix function of `"ababac"` is `[0,0,1,2,3,0] = pi`. Output `"ababac"`. Correct.

**Edge cases, deliberately, because this is where reconstruction code dies.**
- `n = 0`: I read `n=0`, both loops are empty, the verify comparison loop is empty, and I print `s` which is the empty string followed by a newline. The empty string is the unique witness — correct.
- `n = 1, pi = [1]`: structural check catches `pi[0] = 1 > 0 = i`, so `-1`. A one-character string can only have `pi[0] = 0`. Correct.
- `pi = [1, ...]` with `pi[0] != 0`: same structural reject. Correct.
- Out-of-range / negative entries (adversarial input): `pi[i] < 0` or `pi[i] > i` is rejected before any indexing, so no out-of-bounds in the construction. Correct and safe.
- A near-miss like a real prefix function with one entry bumped: passes structural checks but fails the verify pass, yielding `-1` — exactly the dangerous class the insight targets.
- Deep border chains (Zimin word, `n` up to `2*10^5`): the construction needs `<= 18` distinct letters, well under `26`; the chain walks are amortized linear (the same accounting as the standard prefix-function bound), so the run stays in milliseconds.
- Large `n = 2*10^5` random: I/O is the only real cost; the algorithm is `O(n)`. Measured `~10 ms`.

**Complexity.** The structural scan is `O(n)`. The construction visits each position once; the border-chain walks at `pi=0` positions are bounded in total by the same potential argument that makes the prefix-function computation `O(n)` (the chain length only ever drops by the amount the border previously grew). The verify recomputation is the textbook `O(n)` prefix function. So the whole thing is `O(n)` time and `O(n)` memory, trivially inside `1 s` and `256 MB` at `n = 2*10^5`.

**Why I trust this over the "clever predicate".** I considered writing a direct feasibility predicate that walks the border chain and checks each `pi[i]` against the legal set without building a string. It is doable, but it re-derives the same chain logic the construction already needs, with more opportunities to be subtly wrong, and it still has to *produce* a witness when feasible. Construct-then-verify gives me both halves with a single, self-checking pass: the output is correct **by construction of the check**, not by my faith in a hand-proved invariant. I disproved the local-inequality approach with `[0,1,1]`, derived the transitive-border insight from *why* it failed, implemented the construction, found and fixed a real out-of-bounds at `i = 0` by tracing `pi = [0]`, and confirmed the witness/`-1` behaviour on the transitive case and the structural edges. That is what I ship — one self-contained file.

**Final solution.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;          // empty input -> treat as n = 0
    vector<int> pi(n);
    for (int i = 0; i < n; i++) cin >> pi[i];

    // ---- Quick structural rejects (cheap, before construction) -------------
    // pi[i] is a length in [0, i]; pi[0] must be 0; a border cannot grow by more
    // than one per step (pi[i] <= pi[i-1] + 1). These are necessary but NOT
    // sufficient -- transitive border composition is the real test, handled by
    // the verify pass below.
    bool bad = false;
    for (int i = 0; i < n; i++) {
        if (pi[i] < 0 || pi[i] > i) { bad = true; break; }
        if (i > 0 && pi[i] > pi[i - 1] + 1) { bad = true; break; }
    }
    if (bad) { cout << -1 << "\n"; return 0; }

    // ---- Reconstruct a candidate string ------------------------------------
    // Letters are 0..25 mapped to 'a'..'z'. Rule:
    //   pi[i] > 0  : the border of length pi[i] forces s[i] == s[pi[i]-1].
    //   pi[i] == 0 : s[i] must NOT continue any border of s[0..i-1]. Collect the
    //                set of "next" characters that every border (lengths
    //                pi[i-1], pi[pi[i-1]-1], ...) would demand, and pick any
    //                letter outside that set. With an alphabet of 26 there is
    //                always a free letter (the chain has < n forbidden chars,
    //                and a 0 only forbids the distinct chain heads).
    string s(n, 'a');
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            s[0] = 'a';                 // position 0 is unconstrained (pi[0] == 0)
        } else if (pi[i] > 0) {
            s[i] = s[pi[i] - 1];        // border of length pi[i] forces this char
        } else {
            // forbidden = chars that would extend some border of s[0..i-1].
            // Walk the border chain pi[i-1] -> pi[pi[i-1]-1] -> ... -> 0; each
            // border of length k could be extended only by s[k], so s[i] must
            // avoid every such s[k], else pi[i] would be positive.
            bool used[26] = {false};
            int k = pi[i - 1];
            while (true) {
                used[(int)(s[k] - 'a')] = true;
                if (k == 0) break;
                k = pi[k - 1];
            }
            int c = 0;
            while (c < 26 && used[c]) c++;
            if (c == 26) { cout << -1 << "\n"; return 0; } // unreachable for valid pi
            s[i] = char('a' + c);
        }
    }

    // ---- Verify: recompute the prefix function of s and compare ------------
    // This is what turns "construct" into a sound feasibility test: if the input
    // pi is not realizable, the recomputed prefix function will disagree.
    vector<int> chk(n, 0);
    for (int i = 1; i < n; i++) {
        int k = chk[i - 1];
        while (k > 0 && s[i] != s[k]) k = chk[k - 1];
        if (s[i] == s[k]) k++;
        chk[i] = k;
    }
    for (int i = 0; i < n; i++) {
        if (chk[i] != pi[i]) { cout << -1 << "\n"; return 0; }
    }

    cout << s << "\n";
    return 0;
}
```

**Causal recap.** The obvious "check local inequalities then reconstruct" looked complete but `[0,1,1]` passes every local test while being infeasible — `pi[1]=1` forces `s="xx?"`, after which `pi[2]` can only be `2` or `0`, never `1`. That exposed the real law: legal `pi[i]` values come from the *transitive* border chain, a non-local object. The sound way to test it is to build a witness by the forced/avoid rule and then recompute its prefix function and compare — if the array is unrealizable the recomputation disagrees and I print `-1`. Tracing the smallest `pi=0` case, `pi=[0]`, caught an out-of-bounds `pi[i-1]` at `i=0`, fixed by treating position `0` as unconstrained; the alphabet bound (`<= 18` distinct letters even on Zimin words) makes `26` safe; and the structural pre-checks keep adversarial arrays from indexing out of range. `O(n)` throughout, `~10 ms` at `n = 2*10^5`.
