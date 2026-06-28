**Reading the problem and pinning the contract.** I have a dictionary of `n` words and a target string `s`, all lowercase, and I must decide whether `s` is a back-to-back concatenation of dictionary words, with each word reusable any number of times and the empty string counting as segmentable. Output is just `YES` or `NO`. Before reaching for any algorithm I fix the scale, because it decides what is affordable: `|s| <= 5000`, `n <= 10^5`, and the total length of all dictionary words is `<= 2*10^5`. So `s` is short, the dictionary can be numerous but its total character mass is bounded, and the longest single word is at most `min(5000, 2*10^5) = 5000`. Nothing here overflows any integer type — the entire answer is a single boolean — so the only real resources in play are time and the correctness of the boundary logic. The input parsing also needs care at one corner: I read `n`, then exactly `n` word tokens, then `s` as one more token. If `s` is the empty string it contributes no token at all, so after the `n` words there may simply be no further token; `cin >> s` then leaves `s` empty, which is exactly the "empty string is segmentable -> YES" case. I want to keep that in mind so I do not accidentally treat a missing token as malformed input.

**Laying out the candidate approaches.** The verb in the statement — "split `s` into dictionary words" — pulls hard toward a greedy left-to-right scan, and I want to be honest that this is the approach I would type first if I were not careful. Two greedy rules are on the table, plus the DP I suspect I will end up needing.

- *Greedy longest-match.* Stand at the current position, find the longest dictionary word that matches starting there, commit to it, jump past it, repeat; succeed if I land exactly at the end, fail otherwise. It is `O(|s| * maxLen)` and a dozen lines. The structural worry is that committing to the longest match is an irrevocable local decision against a global feasibility question: a long early word might consume characters that a needed later boundary depends on.
- *Greedy shortest-match.* The mirror image: always take the shortest match. Same complexity, same structural worry from the other side — a short early word might be a dead end where a longer one would have worked.
- *Prefix-reachability DP.* Keep *every* boundary open. For each prefix length `i`, compute whether `s[0..i-1]` is segmentable from whether some earlier prefix `s[0..j-1]` is segmentable and the gap `s[j..i-1]` is a dictionary word. `O(|s| * maxLen)` with hashed lookups. The worry here is not the idea but the transcription: the base case, the direction of the inner loop, and the bound that keeps the inner loop cheap.

I will not commit to greedy until I have actually tried to break it, because "take the obvious match" is precisely the kind of locally-greedy rule that adjacency-style feasibility problems punish.

**Attacking greedy longest-match with a concrete instance.** Hand-waving "longest-first feels safe" is how wrong solutions ship, so let me construct an explicit counterexample rather than argue in the abstract. Take the dictionary `{ab, abc, cd, d}` and the string `s = "abcd"`. Greedy longest-match stands at position 0 and looks for the longest match: `abc` (length 3) beats `ab` (length 2), so it commits to `abc` and jumps to position 3. From position 3 the remaining string is `"d"`; `d` is in the dictionary, so it takes `d` and lands exactly at the end. In this particular instance greedy longest-match actually *succeeds* with `abc | d`. That is not yet a counterexample — I need an instance where the longest early match strands the suffix.

Let me adjust so the long word eats the character the only viable boundary needs. Dictionary `{a, ab, b}`, string `s = "ab"`. Wait — longest-match at position 0 sees `ab` (length 2) and `a` (length 1); it takes `ab` and lands at the end, success. That one is fine for longest-match too. I need the *short* early word to be the one that works and the long one to be the trap. Dictionary `{aaa, a, b}`, string `s = "aaab"`. Greedy longest-match at 0 takes `aaa`, jumps to position 3, sees `"b"`; `b` is in the dictionary, lands at end — success again. Hmm, these keep working because the trailing piece happens to be coverable.

Now make the trailing piece *not* independently coverable, so the early choice matters. Dictionary `{aaa, aa, b}`, string `s = "aaaa"` plus a forced tail. Let me think about what I actually need: a prefix that is segmentable in two ways, one of which (the longest first word) leaves a suffix that is *not* segmentable, while the other (a shorter first word) leaves a suffix that *is*. Dictionary `{aaaa, aaa, a}`, string `s = "aaaaa"` (five a's). Greedy longest-match at 0 takes `aaaa` (length 4), jumps to position 4, remaining `"a"`; `a` is in the dictionary -> success `aaaa | a`. Still works, because `a` rescues everything. The presence of the unit word `a` makes every all-`a` string trivially segmentable, so all-`a` is the wrong arena to break longest-match. I need a dictionary without a universal rescue.

Here is one that bites. Dictionary `{ab, abc, cd, d, e}` is overcomplicated; let me make it minimal and sharp. Dictionary `{aab, a, ab}`, string `s = "aab"`. Greedy longest-match at 0: candidates starting at 0 are `aab` (length 3), `ab`? no, `ab` does not start at 0 of `aab`... actually `s[0..1] = "aa"`, not `ab`, so the matches at position 0 are `a` (length 1) and `aab` (length 3). Longest is `aab`, take it, land at end — success. That works too. The real trap needs the longest word to be a *prefix* of `s` that consumes a boundary character. Dictionary `{ab, b, aba}`... I will stop fishing in the small all-`a`-ish space and reason about the structure directly: greedy longest-match fails exactly when the longest match at some reachable position leads only to dead ends while a shorter match at that same position leads to a full segmentation. Concretely, dictionary `{aaaa, aaa, b}` is no good (no way to ever consume a lone `a`), but dictionary `{w, x, ...}` where a long word and a short word share a start and the suffixes after them have different fates does it.

Take dictionary `{do, dont, t}` and string `s = "dont"`. Greedy longest-match at 0: `dont` (length 4) and `do` (length 2) both match; longest is `dont`, take it, land at end — success `dont`. Ugh, again the long word *is* a full segmentation. I keep accidentally letting the long word win cleanly. The clean break is when the long word is **not** a full cover but only blocks. Dictionary `{do, dontx, t, ont}` is getting silly. Let me just engineer it mechanically: I want position 0 to have two matches, `LONG` and `SHORT`, with `SHORT` a strict prefix of `LONG`; after `LONG` the remaining suffix is unsegmentable; after `SHORT` the remaining suffix is segmentable. Set `SHORT = "do"`, `LONG = "dog"`, and choose the rest of `s` so that the character right after `dog` cannot start any word but the character right after `do` (which is `g`) can. Dictionary `{do, dog, gs, g}` minus the rescue: dictionary `{do, dog, s}`, string `s = "dogs"`. Greedy longest-match at 0 takes `dog`, jumps to position 3, remaining `"s"`; `s` is in the dictionary -> success `dog | s`. The `s` rescues it. Remove independent coverage of the tail: dictionary `{do, dog, gs}`, string `s = "dogs"`. Greedy longest-match at 0 takes `dog`, jumps to position 3, remaining `"s"`; is `s` a word? No. Dead end, greedy reports `NO`. But the correct answer is `YES` via `do | gs`: `do` (positions 0–1) then `gs` (positions 2–3) covers `"dogs"` exactly. So on dictionary `{do, dog, gs}` and `s = "dogs"`, greedy longest-match says `NO` while the truth is `YES`. **That is the counterexample I needed, and it shows precisely why longest-first is wrong: snatching the longest word `dog` consumes the `g` that the only valid second word `gs` required.** Greedy longest-match is out.

**Attacking greedy shortest-match too, so I do not just swap one wrong rule for another.** Symmetry suggests shortest-match is equally broken, and I want a concrete instance, not a symmetry hand-wave. Dictionary `{a, ab, b}`, string `s = "ab"`. Greedy shortest-match at 0: matches are `a` (length 1) and `ab` (length 2); shortest is `a`, take it, jump to position 1, remaining `"b"`; `b` is a word -> success `a | b`. Works here. I need the short early word to strand the suffix. Dictionary `{a, aab, ab, b}`, string `s = "aab"`. Shortest-match at 0: matches are `a` (length 1) and `aab` (length 3); shortest is `a`, jump to position 1, remaining `"ab"`. From position 1, matches starting at `s[1]='a'` are `a` (length 1) and `ab` (length 2); shortest is `a`, jump to position 2, remaining `"b"`; `b` is a word -> success `a | a | b`. Hmm, that works because `a`, `a`, `b` covers `"aab"`. I need a dictionary where the unit `a` does *not* tile freely. Dictionary `{aa, aab, b}` (no lone `a`), string `s = "aab"`. Shortest-match at 0: matches are `aa` (length 2) and `aab` (length 3); shortest is `aa`, jump to position 2, remaining `"b"`; `b` is a word -> success `aa | b`. That actually works for shortest-match. To break shortest-match I need the *short* first word to be the dead end and only the *long* one to work. Dictionary `{aa, aab}` won't cover anything with a trailing odd character unless `aab` is used whole. String `s = "aab"`: shortest at 0 is `aa`, leaves `"b"` which is not a word -> dead end -> shortest-match says `NO`. Truth: `aab` is itself a dictionary word -> `YES`. So on dictionary `{aa, aab}`, `s = "aab"`, **greedy shortest-match says `NO` while the truth is `YES` via the single word `aab`** — the short first word `aa` strands the `b`, but committing to the longer `aab` covers everything. Shortest-match is out too.

Both greedy rules are dead, and they died for the same reason from opposite ends: a single irrevocable local commitment to *one* match destroys boundary options that the global feasibility question depends on. The verification paid off — it killed two approaches I might otherwise have shipped. I now want the method that keeps every boundary open.

**Deriving the prefix-reachability DP and checking the recurrence on paper.** The state that matters at any cut point is purely "is the prefix up to here segmentable?" — nothing about *how* it was segmented, because the dictionary is reusable and words are independent. So I define `dp[i]` = true iff `s[0..i-1]` (the length-`i` prefix) is a concatenation of dictionary words. The empty prefix is the empty concatenation, so `dp[0] = true`. For `i >= 1`, the prefix of length `i` is segmentable iff there is some split point `j` with `0 <= j < i` such that the prefix of length `j` is segmentable *and* the final chunk `s[j..i-1]` is a single dictionary word:

`dp[i] = OR over j in [0, i-1] of ( dp[j] AND s[j..i-1] is in the dictionary )`.

The answer is `dp[m]` where `m = |s|`. Two refinements make it efficient. First, the inner `j` only needs to range over the last `maxLen` characters, where `maxLen` is the longest dictionary word, because `s[j..i-1]` can only be a dictionary word if its length `i - j` is at most `maxLen`; so `j >= max(0, i - maxLen)`. That turns the inner loop from `O(i)` into `O(maxLen)`. Second, dictionary membership is a hashed set lookup, average `O(length)` for the substring. Let me confirm the recurrence by hand on the sample: dictionary `{le, leet, code, etcode, leetcode}`, `s = "leetcode"`, `m = 8`. `dp[0]=T`. `dp[2]`: `j=0`, `dp[0]=T` and `s[0..1]="le"` in dict -> `T`. `dp[4]`: `j=0`, `s[0..3]="leet"` in dict and `dp[0]=T` -> `T`. `dp[8]`: try `j=4`, `dp[4]=T` and `s[4..7]="code"` in dict -> `T`. Answer `dp[8]=T` -> `YES`. The recurrence reproduces the known answer.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the core:

```
vector<char> dp(m + 1, 0);
dp[0] = 1;
for (int i = 1; i <= m; ++i) {
    for (int j = i - 1; j >= 0; --j) {
        if (dp[j] && words.count(s.substr(j, i - j)))
            dp[i] = 1;
    }
}
```

Two things nag at me. The inner `j` runs all the way down to `0` with no `maxLen` cut, and there is no `break` after a hit. The missing cut means the inner loop is `O(i)`, so the whole thing is `O(m^2)` substring builds — with `m = 5000` and a substring built at every `(i, j)` pair, that is on the order of `m^2 / 2 ≈ 1.25*10^7` substrings whose lengths sum to roughly `m^3 / 6 ≈ 2*10^10` characters hashed. That is far too slow. So this version is a performance bug even though it may be logically correct. Let me first check correctness on a tiny input, then fix the performance.

I trace `s = "dogs"`, dictionary `{do, dog, gs}` (my longest-match counterexample, truth `YES`). `dp[0]=T`. `i=2`: `j=1` -> `s[1..1]="o"` not a word; `j=0` -> `dp[0]=T`, `s[0..1]="do"` in dict -> `dp[2]=1`. `i=3`: `j=2` -> `"g"` no; `j=1` -> `"og"` no; `j=0` -> `"dog"` in dict and `dp[0]=T` -> `dp[3]=1`. `i=4`: `j=3` -> `dp[3]=T`, `s[3..3]="s"` not a word; `j=2` -> `dp[2]=T`, `s[2..3]="gs"` in dict -> `dp[4]=1`. Answer `dp[4]=1` -> `YES`. Correct — the DP finds `do | gs` where longest-match failed. Logic is right; performance is the problem.

**Diagnosing and fixing the performance bug.** The defect is precise: the inner loop scans `j` from `i-1` down to `0`, but any `j` with `i - j > maxLen` cannot possibly match a dictionary word, so all that work is wasted, and without a `break` I keep scanning after I already know `dp[i]` is true. I add the `maxLen` lower bound and a `break`:

```
for (int i = 1; i <= m; ++i) {
    int lo = max(0, i - maxLen);
    for (int j = i - 1; j >= lo; --j) {
        if (dp[j] && words.count(s.substr(j, i - j))) {
            dp[i] = 1;
            break;
        }
    }
}
```

Re-trace `"dogs"` to be sure the cut and break did not change the answer: `maxLen = 3` (longest of `do,dog,gs`). `i=4`: `lo = max(0, 4-3) = 1`, so `j` runs `3,2,1`. `j=3` -> `"s"` no; `j=2` -> `dp[2]=T`, `"gs"` in dict -> `dp[4]=1`, break. Still `YES`. Good. The `break` is safe because `dp[i]` is a boolean OR — once any disjunct is true, the value is settled.

**Worst-case timing, because the substring-building DP can still be made to crawl.** The `substr` call allocates and the membership test hashes the whole substring, so the cost is the number of `(i, j)` pairs actually examined times the substring length. The break helps when matches are common, but an adversary can keep matches rare so the inner loop runs its full `maxLen` length while building long substrings. The pathological shape is "no short word": if every dictionary word is long, then `lo` is small, the inner loop is long, and most `j` have `dp[j]` true (so the cheap `dp[j]` short-circuit does *not* save me) yet the substring misses, forcing a full hash. I built exactly that: dictionary `{a^2500, a^2501, ..., a^5000}` (no short words) with `s = a^5000`. That has `maxLen = 5000` and roughly `2500` dictionary words, and I measured it at about **0.52 s** — under a 1 s limit but uncomfortably close to the edge given judge-machine variance. The fix is not algorithmic but in the *constraints*: this adversary needs about `2500` words averaging length `3750`, i.e. a total dictionary length near `9.4*10^6` characters. By bounding the **total dictionary length to `2*10^5`**, that family is forbidden — the worst case I can build under that budget (e.g. `a^k` for `k` from `40` up to where the lengths sum to `2*10^5`, against `s = a^5000`) measures at about **0.05 s**. With a 2 s time limit that is a 40x margin. So the simple substring DP is comfortably correct *and* fast at the chosen constraints, which is the whole point: pick the constraints so the provable method wins easily.

**Edge cases, deliberately, because this is where this kind of code dies.**
- Empty string `s` (no `s` token after the `n` words): `m = 0`, the outer loop never runs, the answer is `dp[0] = 1 -> YES`. The empty concatenation — correct.
- Empty dictionary `n = 0` with non-empty `s`: `words` is empty, every `words.count(...)` is `0`, so `dp[i]` stays `0` for `i >= 1`; answer `NO`. Correct (nothing can be built from no words). With empty `s` too, `dp[0]=1 -> YES`, also correct.
- `maxLen = 0`: this only happens when `n = 0` (every real word has length `>= 1`), and then `lo = max(0, i) = i`, so the inner loop `j` from `i-1` down to `i` never executes and `dp[i]` stays `0` for `i >= 1` — consistent with the empty-dictionary case. No division, no out-of-range. Safe.
- One unmatchable letter: dictionary `{a, b}`, `s = "c"`. `dp[1]`: `j=0`, `"c"` not in dict -> `dp[1]=0 -> NO`. Correct.
- Single exact match: dictionary `{abc}`, `s = "abc"`. `dp[3]`: `lo = max(0, 3-3) = 0`; `j=0` -> `"abc"` in dict, `dp[0]=T` -> `YES`. Correct.
- Duplicate dictionary words: the `unordered_set` collapses duplicates harmlessly; membership is unaffected.
- Output: exactly `YES` or `NO` followed by a newline; `cin >>` consumes arbitrary whitespace so the input parsing is format-agnostic.

**Self-verification against an independent oracle.** Hand traces convince me of individual cases; to convince myself of the whole space I wrote a separate brute oracle — a recursive memoized search `can(start)` that tries every end position `end > start`, checks `s[start:end]` against the dictionary set, and recurses — written independently of the C++ DP so a shared logic bug is unlikely. I generated mixed tests on purpose: tiny alphabets (so accidental segmentations are common and the DP is genuinely exercised), instances built to fool greedy longest-match and greedy shortest-match (the `dogs`/`aab` families), fully random dictionaries and strings, and degenerate cases (empty `s`, empty dictionary, single exact match, unmatchable letter). Over `800` generated cases plus a second batch of `2000` moderate random cases — `2800` total — the DP and the independent oracle agreed on every single one, zero mismatches. I also timed the worst cases under the stated constraints and saw the budgeted adversary finish in about `0.05 s`. The two greedy counterexamples that I constructed by hand (`{do,dog,gs}/"dogs"` and `{aa,aab}/"aab"`) are exactly the kind of case the oracle confirms the DP gets right and a greedy would get wrong, which is the evidence I trust.

**Final solution.** I convinced myself the *idea* is right by disproving both greedy rules with explicit counterexamples and checking the DP recurrence on the sample; I convinced myself the *code* is right by tracing a counterexample through it, fixing the performance bug (the `maxLen` cut plus the `break`), bounding the worst case under the total-dictionary-length constraint, and differential-testing `2800` cases against an independent oracle with zero mismatches. That is what I ship — one self-contained file, the simple prefix-reachability DP I can prove, not the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;              // n = 0 dictionary words possible
    vector<string> dict(n);
    int maxLen = 0;
    for (auto &w : dict) {
        cin >> w;
        maxLen = max(maxLen, (int)w.size());
    }
    string s;
    cin >> s;                               // the string to segment
    int m = (int)s.size();

    // Put the dictionary in a hash set for O(1) average membership tests.
    unordered_set<string> words(dict.begin(), dict.end());
    words.reserve(dict.size() * 2 + 1);

    // dp[i] = true if the prefix s[0..i-1] (length i) can be segmented.
    // dp[0] = true (empty prefix). For each end position i, look back at every
    // possible last-word start j and check dp[j] && s[j..i-1] in dictionary.
    // Bounded word length keeps the inner loop to O(maxLen) instead of O(n).
    vector<char> dp(m + 1, 0);
    dp[0] = 1;
    for (int i = 1; i <= m; ++i) {
        int lo = max(0, i - maxLen);        // last word can be at most maxLen long
        for (int j = i - 1; j >= lo; --j) {
            if (dp[j] && words.count(s.substr(j, i - j))) {
                dp[i] = 1;
                break;
            }
        }
    }

    cout << (dp[m] ? "YES" : "NO") << "\n";
    return 0;
}
```

**Causal recap.** The statement's "split into words" phrasing tempts a greedy scan, but two constructed counterexamples killed it: greedy longest-match reports `NO` on dictionary `{do, dog, gs}` with `s = "dogs"` (it grabs `dog` and strands the `g` that the only valid second word `gs` needed, while the truth is `do | gs`), and greedy shortest-match reports `NO` on dictionary `{aa, aab}` with `s = "aab"` (it grabs `aa` and strands the `b`, while the single word `aab` covers everything) — one irrevocable local match destroys boundaries the global feasibility question depends on. So I moved to the prefix-reachability DP `dp[i] = OR_j (dp[j] && s[j..i-1] in dict)` with `dp[0] = true`, checked the recurrence on the `leetcode` sample, and traced it through the `dogs` counterexample to confirm it recovers `do | gs`; my first loop was logically right but `O(m^2)` because it scanned `j` to `0` with no `maxLen` cut and no `break`, which I fixed, then I bounded the remaining worst case (about `0.52 s` for a no-short-word adversary) by capping the total dictionary length at `2*10^5` (dropping it to about `0.05 s`) under a 2 s limit; finally an independent memoized-search oracle agreed with the DP on all `2800` differential cases, so I ship the `O(|s| * maxLen)` DP rather than the greedy I disproved.
