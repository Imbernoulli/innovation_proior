**Reading the problem and pinning the contract.** I have a skyline of `n` towers written as a digit
string `s`, and I must demolish **exactly** `k` of them so the surviving towers — kept in their
original left-to-right order and squeezed together — form a string of length `n - k` that is
lexicographically smallest. Input is `n` and `k` on one line, then the `n`-digit string; I print one
line, the smallest surviving string (empty when `k = n`). I want the scale fixed before I touch an
algorithm: `n <= 2*10^5`, so anything quadratic in the worst case is `4*10^10` operations and will
not survive the 1-second limit. The values are single digits, so there is no integer-overflow story
here — the danger is entirely in the *selection logic*, not the arithmetic. `k` I read as
`long long` defensively (the contract bounds it by `n`, but reading it wide costs nothing and removes
one class of accident). The output is a string, and crucially **no leading zeros are stripped**: I
print every surviving digit exactly as it stands. That is one fewer special case than the classic
"smallest number" phrasing, and I will keep it that way.

**What lexicographic-smallest actually rewards.** Comparison is positional: the first surviving
character is infinitely more important than the second, which dwarfs the third, and so on. So the
real objective is "make the earliest surviving position as small as possible; subject to that, make
the next one as small as possible; ...". A demolition spent on an early, tall tower can promote a
short tower into a very significant slot, which is worth far more than removing a taller tower that
sits late. This observation is the whole problem, and it already makes me suspicious of any approach
that ranks towers by height alone.

**Laying out the candidate approaches.** Two routes are on the table, and I want the one I can
*prove*, not the one that is fastest to type.

- *Value greedy.* Demolish the `k` tallest towers (largest digits), breaking ties somehow. `O(n log n)`
  with a sort, a handful of lines. The structural worry: "tallest" is a property of the digit, but the
  damage a tower does is a property of the digit *and its position*. If those two notions disagree,
  this greedy is wrong.
- *Monotonic stack.* Build the survivors on a stack while scanning left to right. When the tower on
  top of the stack is taller than the tower now arriving and I still have demolitions left, pop the
  top (demolish it) because a shorter tower can take that more-significant slot. `O(n)`, each tower
  pushed and popped at most once. The worries are mechanical: is the pop condition strict (`>`) or
  non-strict (`>=`), and what happens to demolitions I never used.

**Stress-testing the value greedy before committing.** "Remove the tallest" *feels* right, so I will
try to break it with a concrete instance rather than trust the feeling. Take `s = "2102"`, `k = 1`.
The tallest digit is `2`, and there are two of them: index 0 and index 3. The value greedy says
"remove a `2`" but does not, on its own, tell me *which* `2` — and the choice is the entire game.
If a naive tie rule removes the **last** tallest tower (a common, lazy default — "scan from the right
for the max"), I delete index 3 and get `"210"`. If instead I think about it positionally and remove
the **first** `2`, index 0, I get `"102"`, which is strictly smaller (`'1' < '2'` in the leading
slot). So `"102"` beats `"210"`, and the value greedy is only correct if its tie-break happens to
prefer the earliest occurrence of the max — a fragile coincidence, not a principle. Worse, with two
removals the interactions compound: on `s = "12002"`, `k = 2` the positional answer is `"002"`
(demolish the leading `1` and one of the `2`s to expose the two zeros), whereas "remove the two
tallest" with a right-biased tie rule removes the two `2`s and yields `"100"`, which is far larger.
The verification paid off: the value greedy's correctness depends entirely on a tie rule it does not
naturally possess, and even a "remove earliest max" variant is not obviously right under multiple
removals. I discard it. The lesson it taught me is the design principle for the stack: I should be
demolishing the **earliest tower that a shorter successor can improve upon**, not the globally
tallest one.

**Deriving the monotonic stack and arguing it is correct.** I scan the towers left to right, keeping
the survivors-so-far on a stack `st` (a string I push/pop from the back). I carry a `budget` of
remaining demolitions, initially `k`. When tower `c` arrives, as long as `budget > 0` and the top of
the stack is **strictly taller** than `c`, I pop the top and spend one demolition. Then I push `c`.

Why is this optimal? Consider the leftmost position in the final answer. I want the smallest digit
that can legally sit there. As I scan, the stack holds a prefix of survivors; if a later, shorter
tower `c` arrives while a taller tower `t` sits above it on the stack and I can still afford a
demolition, then deleting `t` strictly improves the answer at `t`'s (significant) position — `c` or
something at least as small now occupies that slot, and nothing earlier is harmed because everything
below `t` on the stack is `<= t`'s neighbours by the same invariant. Each tower is pushed once and
popped at most once, so the work is `O(n)`. The stack is, by construction, non-decreasing from bottom
to top at all times, which is the monotonic invariant that makes the greedy local decision globally
correct.

**Two mechanical questions the derivation leaves open.** First, strict vs non-strict pop. If the top
equals the arriving digit, popping it spends a demolition without improving the leading slot (same
digit takes the place), and that wasted budget could have bought a real improvement later. So the
condition must be **strict** `>`; popping on `>=` throws budget away. Second, leftover budget: if the
skyline is non-decreasing, no pop ever fires and I finish the scan with `budget > 0`. Those removals
still must happen (exactly `k`), and on a non-decreasing tail the smallest result is obtained by
dropping from the **end** — removing later (less significant) towers. So after the scan I pop the tail
`budget` more times. Both of these I will get wrong on the first draft if I am not careful, so I will
trace them.

**Sanity-checking the recurrence on the stated sample.** `s = "1432219"`, `k = 3`, expected `"1219"`.
Start `st = ""`, `budget = 3`.
- `'1'`: stack empty, push. `st = "1"`.
- `'4'`: top `'1' < '4'`, no pop. push. `st = "14"`.
- `'3'`: top `'4' > '3'`, budget 3→2, pop `'4'`. now top `'1' < '3'`, stop. push. `st = "13"`.
- `'2'`: top `'3' > '2'`, budget 2→1, pop `'3'`. top `'1' < '2'`, stop. push. `st = "12"`.
- `'2'`: top `'2'` is **not** `> '2'`, no pop (strictness matters here). push. `st = "122"`.
- `'1'`: top `'2' > '1'`, budget 1→0, pop. budget now 0, stop. push. `st = "121"`.
- `'9'`: budget 0, no pop. push. `st = "1219"`.
Final `budget = 0`, answer `"1219"`. Matches. And I note the strict comparison was load-bearing at
the second `'2'`: a non-strict `>=` would have popped it, spent my last budget, and produced the wrong
string.

**First implementation — then immediately a trace, because clean reasoning transcribes dirty.** My
first cut of the core loop:

```
string st; long long budget = k;
for (int i = 0; i < n; i++) {
    char c = s[i];
    while (budget > 0 && !st.empty() && st.back() >= c) {  // <-- suspicious
        st.pop_back();
        budget--;
    }
    st.push_back(c);
}
// (no tail handling yet)
cout << st << "\n";
```

I deliberately wrote `>=` to see whether my own argument about strictness holds up under a trace, and
I left out the tail drain to see it fail. Smallest input that exposes the equality question:
`s = "112"`, `k = 1`. The answer should be `"11"` — the only useful removal is deleting the `'2'`,
leaving the two ones. Trace with `>=`: `budget = 1`.
- `'1'`: empty, push. `st = "1"`.
- `'1'`: top `'1' >= '1'` is **true**, so pop, budget 1→0. push `'1'`. `st = "1"`.
- `'2'`: budget 0, no pop. push. `st = "12"`.
Output `"12"`.

**Diagnosing the first bug.** `"12"` is wrong; the answer is `"11"`. The defect is exactly the one my
derivation warned about: `>=` popped an equal tower (the first `'1'`) and burned my single demolition
for *no improvement* — the slot still holds a `'1'` — so when the genuinely removable `'2'` arrives I
have no budget left. Equality must not trigger a pop. I change `>=` to `>`. Re-trace `"112"`, `k = 1`
with `>`: `'1'` push → `"1"`; `'1'` top `'1'` not `> '1'`, push → `"11"`; `'2'` top `'1'` not `> '2'`,
push → `"112"`, `budget = 1` left over. Output... `"112"` — length 3, but I must remove one. That
surfaces the *second* bug.

**Diagnosing the second bug (leftover budget).** With the strict condition fixed, a non-decreasing
input like `"112"` (or `"12345"`) finishes the scan with `budget > 0` and I never spend it, so I
output `n` characters instead of `n - k`. That is both the wrong length and, in spirit, "demolish
fewer than `k`". The fix is the tail drain I had reasoned about but not coded: after the loop, while
`budget > 0`, pop from the back. On a non-decreasing remainder the back holds the least significant
towers, so dropping them is optimal. I add:

```
while (budget > 0) { st.pop_back(); budget--; }
```

**Re-verifying both fixes on the cases that broke.** With `>` **and** the tail drain:
- `"112"`, `k = 1`: scan leaves `st = "112"`, `budget = 1`; drain pops one from the back → `"11"`.
  Correct.
- `"12345"`, `k = 2`: scan never pops (strictly increasing), `st = "12345"`, `budget = 2`; drain pops
  two → `"123"`. Correct — the smallest length-3 subsequence of an increasing string is its prefix.
- Re-run the sample `"1432219"`, `k = 3`: as traced above, `budget` hits 0 mid-scan, the drain does
  nothing, answer `"1219"`. Correct.
Both originally-failing cases now pass, and they pass *for the reasons I diagnosed* — equality no
longer wastes budget, and leftover budget is forced onto the least significant tail. That match
between predicted cause and observed fix is the evidence I trust.

**Edge cases, deliberately, because this is where selection code dies.**
- `k = 0`: `budget = 0`, the inner `while` never fires, the drain never fires, I push every tower and
  print `s` unchanged. Correct.
- `k = n`: every tower must go. Trace `"9081"`, `k = 4`. `'9'` push → `"9"`. `'0'`: top `'9' > '0'`,
  pop (budget 4→3); stack empty, push → `"0"`. `'8'`: top `'0'` not `> '8'`, push → `"08"`. `'1'`:
  top `'8' > '1'`, pop (3→2); top `'0'` not `> '1'`, push → `"01"`. Scan ends `st = "01"`,
  `budget = 2`; drain pops both → `""`. I print an empty line. Correct (length `n - k = 0`).
- All-equal, `"00000"`, `k = 2`: no pop ever fires (`'0'` is never `> '0'`), `st = "00000"`,
  `budget = 2`, drain pops two → `"000"`. Correct.
- Strictly decreasing, `"9531"`, `k = 2`: `'9'` push; `'5'` pop `'9'` (budget→1) push `'5'`; `'3'`
  pop `'5'` (budget→0) push `'3'`; `'1'` budget 0, push → `"31"`. Drain idle. Answer `"31"`. Correct
  — the two front (most significant) towers were demolished.
- The positional counterexample that killed the value greedy, `"2102"`, `k = 1`: `'2'` push; `'1'`
  top `'2' > '1'`, pop (budget→0) push `'1'` → `"1"`; `'0'` budget 0 push → `"10"`; `'2'` push →
  `"102"`. Answer `"102"`, exactly the positional optimum the greedy missed. The stack found it for
  free.
- Output format: one line, the string then `"\n"`; the empty-result case prints just the newline,
  which the contract allows.

**Performance check.** Each character is pushed exactly once and popped at most once across the whole
run (the drain only removes characters that were pushed), so the total stack operations are `O(n)`.
Reading the string is `O(n)`. At `n = 2*10^5` this is trivially inside 1 second; I use a `string` for
`st` and `reserve(n)` to avoid reallocation churn. No sorting, no quadratic rescans.

**Final solution.** I convinced myself the *idea* is right by disproving the value greedy with a
positional counterexample and by hand-checking the stack's invariant on the sample, and I convinced
myself the *code* is right by tracing the two failing cases to precise causes — equality wasting
budget, and leftover budget never spent — and re-verifying each fix and every corner. That is what I
ship: one self-contained file, the `O(n)` monotonic stack I can defend rather than the greedy I
broke.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;
    string s;
    cin >> s;

    // Monotonic (non-decreasing) stack: while we still have removals left and the
    // top of the stack is strictly greater than the current character, pop it.
    // This greedily fixes the earliest position where a smaller character can
    // take a more significant slot.
    string st;
    st.reserve(n);
    long long budget = k;
    for (int i = 0; i < n; i++) {
        char c = s[i];
        while (budget > 0 && !st.empty() && st.back() > c) {
            st.pop_back();
            budget--;
        }
        st.push_back(c);
    }
    // If removals remain (string was non-decreasing), drop from the tail.
    while (budget > 0) {
        st.pop_back();
        budget--;
    }

    cout << st << "\n";
    return 0;
}
```

**Causal recap.** The value greedy "demolish the tallest towers" looked right but a positional
counterexample (`"2102"`, `k = 1`: removing the *last* tallest gives `"210"` while removing the
*earliest* tall tower gives the smaller `"102"`, and `"12002"`, `k = 2` shows the gap widening to
`"100"` vs `"002"`) revealed that damage is positional, not value-based — so I moved to a
left-to-right monotonic stack that pops a surviving tower whenever a shorter one arrives and budget
remains, demolishing the *earliest improvable* tower instead of the globally tallest. Two transcription
bugs surfaced under tracing: a non-strict `>=` pop wasted my budget on equal towers (`"112"`, `k = 1`
returning `"12"` instead of `"11"`), fixed by requiring strict `>`; and forgetting that a
non-decreasing skyline ends the scan with budget unspent (`"12345"`, `k = 2` returning the full
string) forced me to add a tail drain that removes the least significant towers. With strict pops and
the tail drain, the stack is `O(n)` and provably optimal, and it closes the `k = 0`, `k = n`,
all-equal, strictly decreasing, and empty-output corners.
