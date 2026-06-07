OK, let me think this through from scratch. I'm searching a long text for a word — say I'm looking for `dikran`. The obvious thing is to start at every position in the text and check character by character: is it a `d`? yes. is the next an `i`? yes. is it a `k`? no — this was the word `direction`. So I give up, move one position to the right, and start over: is *this* a `d`? But wait — I already looked at that character a moment ago. When `dik` failed against `dir`, I had already read `d i r`, and I'm about to ask "is the next character a `d`?" of a character I *know* is an `i`. It's not a `d`, so I might as well skip ahead further before testing again. The naive scan throws away everything it learned during the partial match.

Let me make the cost of that waste precise. Aligning the pattern at every text start and comparing forward until a mismatch is, in the worst case, about `n` starts times `m` comparisons each — `O(nm)`. The truly bad input is something like pattern `a^k b` against text `a^N b`: at every one of the `N` alignments I match a long run of `a`s and then fail on the final `b`, redoing almost the whole pattern each time. And there's a second, more practical wound. To re-try at the next start I have to *back the text pointer up* — re-read characters I already passed. If the text is streaming in from a file or a buffer in an editor, backing up means I have to hold onto those characters, and the buffering bookkeeping for that is genuinely annoying. What I really want is a scan that reads the text strictly left to right, **never moving the text pointer backward**, and still finds every match. If I could do that, the buffering problem evaporates and the `O(nm)` re-reading goes with it.

So the goal is sharp: one forward pass over the text, no backup. The question is whether that's even possible, and if so how cheaply. Here my hand-intuition stalls. I can *feel* that a partial match tells me something — when I matched `abc` and then hit `x ≠` the fourth pattern char, I now know the last four text characters were exactly `a b c x` — but I don't have a systematic way to convert "I matched `j−1` characters and then failed" into "here's exactly how far I can slide the pattern, and I never need to look back at the text." Let me hold that thought.

Now, a completely different thing has been nagging at me, and I want to bring it alongside. Steve Cook proved a theorem that genuinely surprised me. Take a very limited kind of machine — a two-way deterministic pushdown automaton: a finite-state control, a stack, and a single read-only head that can move both left and right on the input. Such a machine can be absurdly slow; it can run for an exponential number of steps before it halts, because it can wander back and forth building up and tearing down enormous stacks. Cook's theorem says: *no matter how slow* that pushdown machine is, if it recognizes a language at all, then there is a way to recognize that same language on an ordinary random-access computer in `O(n)` time — linear. A possibly-exponential machine, simulated in linear time. That's a strange and strong claim.

And it connects to a problem I can't crack by hand. Cook and others had been looking at palindrome problems. Chester showed that the set of strings *beginning with an even palindrome* can be recognized by one of these two-way pushdown machines. So by Cook's theorem there's a linear-time way to recognize it on a real computer. But I sit down to recognize "begins with an even palindrome" — or concatenations of even palindromes — on a normal machine, and I cannot see how to do it in less than about `n²` steps. To check whether a chunk is a palindrome I want to compare it against its own reversal, and finding the right split points seems to force me to re-scan. I'm a decent programmer and here's a theorem flatly telling me a *linear* method exists, and I can't think of it. That's the part that stings, and it's also the clue: the theorem isn't just an existence statement, it's *constructive*. Cook's proof actually builds the linear RAM procedure out of the pushdown machine. So if I want to know what the fast method *is*, I shouldn't keep guessing — I should take the specific pushdown automaton for the palindrome problem, run it through Cook's construction step by step, and watch what the construction produces. Then distill off the mechanism: figure out *why* the result is linear, in plain terms I can re-use.

I go to the blackboard and trace Cook's construction on the palindrome automaton carefully.

What is Cook's construction even tracking? The pushdown machine at any instant has a state, a head position on the input, and a stack. The full instantaneous description includes the entire stack, and there are exponentially many possible stacks — that's the source of the exponential time. But Cook's insight is to look at the **surface configuration**: just the triple `(state, head position, top-of-stack symbol)`. The symbols buried under the top don't affect what the machine does *next*; only when the top gets popped does the next symbol down become relevant. The number of distinct surface configurations is small — states and stack symbols are fixed, head position ranges over the `n` input cells — so there are only `O(n)` of them.

Fix a surface configuration `c` sitting at some stack height `h`. Run the machine forward from `c`. The stack will go up and down, but eventually — in a terminating computation — it drops *below* height `h` for the first time. Call the configuration at that moment the **terminator** of `c`. The terminator of `c` depends only on `c`, not on whatever is buried below in the stack, because the machine cannot touch those buried symbols until it has popped down past them, and the terminator is the very moment it first pops below `c`. Terminators also compose. If from `c` the machine first pushes, reaching some `c'`, then to find where `c` eventually pops below its height I should find the terminator of `c'` (where the pushed material gets cleared), and from there continue. The terminator of a configuration that pushes is computed from the terminator of the configuration after the push. So the whole forward simulation can be expressed as: compute terminators recursively, and remember each terminator once I've computed it, in a table indexed by the surface configuration. Memoize `T[c]`. Because there are only `O(n)` surface configurations, and each terminator is computed once and then looked up forever after, the total work is `O(n)`. The exponential blow-up came from the machine re-deriving the same sub-computations over and over as it shuffled its stack; sharing the terminators through the table is exactly what kills the repetition. *That* is why Cook's simulation is linear — the sharing of repeated sub-computations via a precomputed table.

Now I stare at this for the specific palindrome automaton and ask: when I strip away the pushdown-machine scaffolding, what is the table actually remembering? In the palindrome case, the useful residue of a failed attempt is not the whole history of the head moving back and forth. It is a compact surface fact: how much of the current string comparison has survived, and where the next viable comparison can resume after the doomed part has been popped away. That is exactly the kind of fact Cook's terminator table stores. If I generalize this from palindromes to the simpler task "find the longest prefix of one string that occurs in another," the meaningful state at a failed comparison is just the number `j`: "I had matched `j−1` characters of the pattern and then the next one disagreed with the text." Everything needed to recover from that mismatch is a function of `j` and of the pattern alone. It does **not** depend on the unread text. The terminator-table idea, specialized to this string-comparison situation, collapses from "indexed by surface configuration" down to "indexed by the matched-prefix length `j`."

The surface-configuration table has collapsed to the table I want: for each pattern position `j`, a single number telling me, when a mismatch occurs after matching `j−1` characters, what pattern position to resume the comparison at — equivalently, how far to slide the pattern — **without touching the text pointer**. The would-be backtracking is paid for once, off the pattern, and stored in a small table. Once that table exists, the matcher can keep the text head monotone: mismatch recovery changes only the pattern pointer. The two threads — "I want a forward-only matcher" and "Cook's construction shows how repeated failed subcomputations can be shared" — are the same thread. Let me build the table directly, in string terms, and forget the automaton now that I have the mechanism.

Set up the matcher concretely. Keep a text pointer `k` and a pattern pointer `j`; the pattern is aligned so that its first character sits at text position `k − j + 1`. When `text[k] = pattern[j]`, advance both: `k := k+1, j := j+1`. When they disagree, I want to slide the pattern right and resume comparing — but I must *not* move `k` back. Sliding the pattern and resuming means: there's some shorter prefix of the pattern that still matches the text ending just before `k`, and I continue from the next pattern character. Let me name the table `next[j]`: the pattern position to compare against `text[k]` after a mismatch at `pattern[j]`. So the inner recovery is just `j := next[j]`, repeated until either the characters match or `j` falls to `0` (meaning no prefix survives — slide the pattern entirely past and advance the text). The matching loop:

```
j := k := 1
while k ≤ n do
    while j > 0 and text[k] ≠ pattern[j] do j := next[j]
    k := k + 1; j := j + 1
    if j = m + 1 then
        record k − m
        j := f[m+1]
```

Notice what this loop does *not* do: it never decreases `k`. `k` only ever increments. The recovery on mismatch happens entirely by shrinking `j` through the `next` table. That's the no-backup property, fallen straight out.

Let me pin down exactly what `next[j]` must be for this to be correct. When I'm about to set `j := next[j]`, I know `j > 0`, and I know the last few text characters, up to and including `text[k]`, are
```
pattern[1] … pattern[j−1]  x        (x = text[k], x ≠ pattern[j])
```
because I'd matched the first `j−1` pattern characters before failing on the `j`-th. I want the *smallest* slide that could still lead to a match, which is the same thing as the largest surviving prefix. So I want the largest `i < j` such that the last `i−1` characters before `x` already line up with the start of the pattern:
```
pattern[1] … pattern[i−1]  =  pattern[j−i+1] … pattern[j−1].
```
In other words, `pattern[1..i−1]` is both a prefix and a suffix of `pattern[1..j−1]` — a *border*. A border of length `i−1` means the shift is `j−i`; the larger `i` is, the less I slide, and the less risk I have of skipping a possible match. If there is no usable nonempty border, `next[j] = 0` and I slide the pattern past the current text character.

But there's a sharpening I shouldn't miss. Suppose the longest border has length `i−1`, so I'd resume comparing `text[k]` against `pattern[i]`. If `pattern[i]` happens to *equal* `pattern[j]` — the very character that just mismatched `text[k]` — then comparing `text[k]` against `pattern[i]` is guaranteed to fail too, since `text[k] ≠ pattern[j] = pattern[i]`. That's a wasted comparison. So I should *not* resume at such an `i`; I should demand `pattern[i] ≠ pattern[j]` in the definition, and if the longest border's `pattern[i]` equals `pattern[j]`, skip past it to the next-longest. Let me define the plain border function first and then apply this refinement.

Let `f[j]` = the length-plus-one position of the longest proper border of `pattern[1..j−1]`; precisely, the largest `i < j` with `pattern[1..i−1] = pattern[j−i+1..j−1]`. This holds vacuously for `i = 1`, so `f[j] ≥ 1` for `j > 1`, and set `f[1] = 0` by convention. Set `next[1] = 0`; for `j > 1`, `next[j]` is the same thing but with the inequality refinement:
```
next[j] = f[j]        if pattern[j] ≠ pattern[f[j]]
        = next[f[j]]  if pattern[j] = pattern[f[j]].
```
Equivalently, `next[j]` is the largest `i < j` with `pattern[1..i−1] = pattern[j−i+1..j−1]` and `pattern[i] ≠ pattern[j]`, or `0` if no such `i` exists. The recursion in the second line is right because if `pattern[j] = pattern[f[j]]`, resuming at `f[j]` is the wasted comparison, so I take whatever I *would* have taken after a mismatch at `f[j]`, namely `next[f[j]]`, which is already computed.

Now, how do I compute `f` and `next` in the first place? Look at the defining condition for `f[j]`: I'm searching for the longest prefix of the pattern that matches a suffix of `pattern[1..j−1]`. That is *exactly* the matching problem again — matching the pattern against itself. I can run the same loop with the text replaced by the pattern. Imagine two copies of the pattern, an upper and a lower, the upper sliding rightward against the lower; the borders are the overlaps. So the preprocessing reuses the matching machinery on `text = pattern`. Concretely, I carry a candidate `t` for `f[j]`. If `t > 0` and `pattern[j]` disagrees with `pattern[t]`, I slide through `t := next[t]` until either `t = 0` or the candidate can extend. Then I increment `t`, so it becomes `f[j+1]`, and I apply the same guaranteed-recomparison refinement to decide `next[j+1]`. The program:

```
t := 0; next[1] := 0
for j := 1 to m−1 do      # t = f[j] going in
    while t > 0 and pattern[j] ≠ pattern[t] do t := next[t]
    t := t + 1            # now t = f[j+1]
    if pattern[j+1] = pattern[t] then next[j+1] := next[t]
    else next[j+1] := t
```

I never even need to store `f` separately — I fold the refinement in as I go, using `next[t]` to slide. This is `O(m)`: the inner `t := next[t]` always shifts the upper copy of the pattern strictly right, so over the whole preprocessing it runs at most `m` times in total. (Cleaner amortized argument: `t` starts at `0`, increases by `1` exactly `m−1` times, and stays `≥ 0`; since `t := next[t]` strictly decreases `t`, it can fire at most `m−1` times overall.)

Let me check the table on a concrete pattern so I trust it. Take `abcabcacab`, positions `1..10`. The border positions are `f[1..10] = (0, 1, 1, 1, 2, 3, 4, 5, 1, 2)`, and the refined resume positions are `next[1..10] = (0, 1, 1, 0, 1, 1, 0, 5, 0, 1)`. Read off `next`: e.g. at `j = 4` the longest border of `abc` is empty so `f[4] = 1`, and `pattern[1] = a = pattern[4]`, so the refinement kicks in and `next[4] = next[1] = 0` — meaning a mismatch at the fourth character slides the pattern four places, never re-testing `text[k]` against another `a`. That matches the hand-intuition I started with: after `abc` then `x ≠ a`, slide all the way. Good — the table encodes exactly the "skip ahead because you already know that character" insight, computed mechanically.

Now the same amortized argument bounds the *matching* loop. The text pointer `k` advances at most `n` times. The assignment `j := next[j]` always replaces `j` by a smaller nonnegative value, while `j` is increased by `1` only when `k` advances. Charge every strict decrease of `j` to a previous increase of `j`; there are at most `n` such increases, so the inner-loop assignment is performed at most `n` times over the whole scan. Matching is `O(n)`, preprocessing is `O(m)`, the grand total is `O(m + n)`, the text pointer never moves backward, and I only need the pattern and its `O(m)` table in memory while the text streams past. That's everything I asked for.

I should convince myself the matching loop is actually *correct*, not just fast — that `next[j]` never makes me skip a real match. The invariant to maintain: let `p = k − j` (the text position just before the pattern's first character in the current alignment). Then `text[p+i] = pattern[i]` for `1 ≤ i < j` (the first `j−1` pattern characters are matched), and for every earlier alignment `0 ≤ t < p` there is some position where `text` and `pattern` already disagreed (no full match was skipped to the left of `p`). When I match, `k` and `j` both advance and `p` is unchanged, so the matched-prefix part of the invariant extends by one. When I set `j := next[j]`, the new alignment starts at `p' = k - next[j]`, which is to the right of `p` because `next[j] < j`. Every start position between `p` and `p'` is ruled out by the information already in hand: either its overlap is not a border of `pattern[1..j−1]`, so it would disagree with one of the matched text characters before `k`, or it is a border whose next pattern character equals `pattern[j]`, so it would disagree with the same current character `text[k]` because `text[k] ≠ pattern[j]`. The remaining candidate is exactly the next border not already killed by that mismatch; if it still disagrees with `text[k]`, the same argument applies again. So the "no match skipped" part stays intact. When `j = m + 1`, the invariant says the whole pattern has just matched ending at `k−1`; recording `k−m` is sound, and resetting `j` to the longest border position `f[m+1]` is exactly the overlapping-match case of the same invariant. The invariant holds across all moves.

There's a subtlety worth checking: I used the *refined* `next` (with `pattern[i] ≠ pattern[j]`) in matching, but should the preprocessing's internal sliding also use `next`, or the plain `f`? For correctness of the resulting matcher, either works — the algorithm runs in linear time even if I used `f[j]` instead of `next[j]` in the main loop. The difference is the per-character delay. Consider pattern `a^m` against text `a^{m−1} b …`: with the plain `f`, where `f[j] = j−1`, a single mismatch at the end triggers `j := f[j]` cascading down `m, m−1, …, 1` — that's `m` slides and `m−1` redundant re-comparisons of the *same* text character before `k` finally advances. The refined `next` collapses that to one step, because `pattern[i] = pattern[j]` all along the chain and the refinement short-circuits it. So `next` is what keeps the work *between two consecutive text inputs* small — it matters if I care about real-time, character-at-a-time reading, not just total throughput.

How small is that per-character delay in the worst case? Let me push on it, because it's a clean question: how many times can `j := next[j]` fire while a single text character is being scanned? Each firing slides the pattern by some amount, and consecutive slides correspond to nested borders of the matched prefix — i.e. to a string having several periods at once. Two periods of one string can't be too small together: if a string of length `ℓ` has periods `p` and `q` with `p + q ≤ ℓ` (roughly), it also has period `gcd(p,q)` — the Fine–Wilf phenomenon — which forces structure. Working out the worst case: I want a pattern that admits the longest possible chain of successive slides, which means a prefix with as many nested distinct periods as possible packed into as short a length as possible. The **Fibonacci strings** do exactly this. Define `b₁ = b, b₂ = a, bₙ = bₙ₋₁ bₙ₋₂`: so `ab, aba, abaab, abaababa, …`, with `|bₙ| = Fₙ` the `n`-th Fibonacci number. These have a near-commutative self-overlap property — changing the last two characters of `bₙ₋₁ bₙ₋₂` gives `bₙ₋₂ bₙ₋₁` — which makes their `next` values form a long descending chain: a mismatch can send `j` through `next` values like `20, 12, 7, 4, 2, 1, 0` — successive Fibonacci numbers (minus one). Since `Fₖ ≈ φᵏ/√5` with `φ = (1+√5)/2` the golden ratio, the number of consecutive `next` steps on a pattern of length `m` is about `log_φ m`. And the Fibonacci pattern shows this is tight: the maximum number of consecutive `j := next[j]` operations while scanning one text character is `Θ(log_φ m)` — at most `1 + log_φ m`. So between two successive single-character inputs only `O(log m)` work elapses; a real-time reader is fine. (With the unrefined `f`, that bound fails — recall the `a^m` example does `m` steps at one character.)

I have the scan; now I need the executable version. I'll switch to `0`-based indexing because it is cleaner in a real implementation, and I'll fold the "slide all the way past, advance text" case (`j = 0` in the `1`-based version) into a sentinel: let the table hold `−1` at the front, so a mismatch that drives the pattern pointer below `0` signals "advance the text pointer and reset," with no special-case branch. The table built this way is the refined `next` for mismatches, and the extra final slot is the restart border length after a complete match so overlapping matches are still found.

```python
def preprocess(W):
    # Build the failure / "next" table from the pattern alone (O(m)).
    # For k < len(W), T[k] resumes after a mismatch at W[k] without
    # moving the text pointer. T[len(W)] restarts after a full match.
    # T[0] = -1 is the sentinel for "no prefix survives."
    T = [0] * (len(W) + 1)
    T[0] = -1
    pos, cnd = 1, 0          # cnd tracks the current border length (the f[j] role)
    while pos < len(W):
        if W[pos] == W[cnd]:
            # border extends, AND the next chars agree, so resuming at cnd
            # would just re-mismatch the same text char — short-circuit it
            # (this is the pattern[i] != pattern[j] refinement).
            T[pos] = T[cnd]
        else:
            T[pos] = cnd
            # slide the pattern against itself via the table already built
            while cnd >= 0 and W[pos] != W[cnd]:
                cnd = T[cnd]
        pos += 1
        cnd += 1
    T[pos] = cnd
    return T


def search(S, W):
    T = preprocess(W)
    matches = []
    k = 0   # text pointer — only ever advances; never backs up
    j = 0   # pattern pointer
    while k < len(S):
        if W[j] == S[k]:
            k += 1
            j += 1
            if j == len(W):                 # full match ends just before k
                matches.append(k - j)
                j = T[j]                     # resume to find further matches
        else:
            j = T[j]                         # slide the pattern, keep k fixed
            if j < 0:                        # sentinel: nothing survives
                k += 1
                j += 1
    return matches
```

The causal chain, end to end: the naive scanner wastes work and, worse, backs the text pointer up, which an editor's buffering makes painful — so I want a strictly forward, no-backup scan, and I can feel that a partial match already tells me the characters just read, but I have no systematic way to exploit it. Cook's theorem insists a linear method *exists* for problems I can phrase as a two-way pushdown automaton, including a palindrome problem I can't crack by hand — so I trace his construction and find its engine is memoizing each surface configuration's terminator, sharing repeated sub-computations through a table, which is why a possibly-exponential machine simulates in linear time. Distilling that to the matching automaton, the only thing the table needs to remember is, for each matched-prefix length `j`, where to resume after a mismatch — a function of the pattern alone — so I precompute a `next` table off the pattern, with the `pattern[i] ≠ pattern[j]` refinement to avoid re-comparisons; building it is the same matching algorithm run on the pattern against itself, in `O(m)`; the matcher then makes only the text pointer advance, recovering on mismatch by shrinking `j` through `next`, total `O(m + n)`, never backing up, `O(m)` memory — and the worst-case per-character delay is a tight `Θ(log_φ m)`, witnessed by Fibonacci strings.
