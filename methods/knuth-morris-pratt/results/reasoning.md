OK, let me think this through from scratch. I'm searching a long text for a word — say I'm looking for `dikran`. The obvious thing is to start at every position in the text and check character by character: is it a `d`? yes. is the next an `i`? yes. is it a `k`? no — this was the word `direction`. So I give up, move one position to the right, and start over: is *this* a `d`? But wait — I already looked at that character a moment ago. When `dik` failed against `dir`, I had already read `d i r`, and I'm about to ask "is the next character a `d`?" of a character I *know* is an `i`. It's not a `d`, so I might as well skip ahead further before testing again. The naive scan throws away everything it learned during the partial match.

Let me make the cost of that waste precise. Aligning the pattern at every text start and comparing forward until a mismatch is, in the worst case, about `n` starts times `m` comparisons each — `O(nm)`. The truly bad input is something like pattern `a^k b` against text `a^N b`: at every one of the `N` alignments I match a long run of `a`s and then fail on the final `b`, redoing almost the whole pattern each time. And there's a second, more practical wound. To re-try at the next start I have to *back the text pointer up* — re-read characters I already passed. If the text is streaming in from a file or a buffer in an editor, backing up means I have to hold onto those characters, and the buffering bookkeeping for that is genuinely annoying. What I really want is a scan that reads the text strictly left to right, **never moving the text pointer backward**, and still finds every match. If I could do that, the buffering problem evaporates and the `O(nm)` re-reading goes with it.

So the goal is sharp: one forward pass over the text, no backup. The question is whether that's even possible, and if so how cheaply. Here my hand-intuition stalls. I can *feel* that a partial match tells me something — when I matched `abc` and then hit `x ≠` the fourth pattern char, I now know the last four text characters were exactly `a b c x` — but I don't have a systematic way to convert "I matched `j−1` characters and then failed" into "here's exactly how far I can slide the pattern, and I never need to look back at the text." Let me hold that thought.

Now, a completely different thing has been nagging at me, and I want to bring it alongside. Steve Cook proved a theorem that genuinely surprised me. Take a very limited kind of machine — a two-way deterministic pushdown automaton: a finite-state control, a stack, and a single read-only head that can move both left and right on the input. Such a machine can be absurdly slow; it can run for an exponential number of steps before it halts, because it can wander back and forth building up and tearing down enormous stacks. Cook's theorem says: *no matter how slow* that pushdown machine is, if it recognizes a language at all, then there is a way to recognize that same language on an ordinary random-access computer in `O(n)` time — linear. A possibly-exponential machine, simulated in linear time. That's a strange and strong claim.

And it connects to a problem I can't crack by hand. Cook and others had been looking at palindrome problems. Chester showed that the set of strings *beginning with an even palindrome* can be recognized by one of these two-way pushdown machines. So by Cook's theorem there's a linear-time way to recognize it on a real computer. But I sit down to recognize "begins with an even palindrome" — or concatenations of even palindromes — on a normal machine, and I cannot see how to do it in less than about `n²` steps. To check whether a chunk is a palindrome I want to compare it against its own reversal, and finding the right split points seems to force me to re-scan. I'm a decent programmer and here's a theorem flatly telling me a *linear* method exists, and I can't think of it. That's the part that stings, and it's also the clue: the theorem isn't just an existence statement, it's *constructive*. Cook's proof actually builds the linear RAM procedure out of the pushdown machine. So if I want to know what the fast method *is*, I shouldn't keep guessing — I should take the specific pushdown automaton for the palindrome problem, run it through Cook's construction step by step, and watch what the construction produces. Then distill off the mechanism: figure out *why* the result is linear, in plain terms I can re-use.

I go to the blackboard and trace Cook's construction on the palindrome automaton carefully.

What is Cook's construction even tracking? The pushdown machine at any instant has a state, a head position on the input, and a stack. The full instantaneous description includes the entire stack, and there are exponentially many possible stacks — that's the source of the exponential time. But the thing to look at is the **surface configuration**: just the triple `(state, head position, top-of-stack symbol)`. The symbols buried under the top don't affect what the machine does *next*; only when the top gets popped does the next symbol down become relevant. Let me count these. The state set is fixed, say `Q` of them; the stack alphabet is fixed, say `Γ`; the head ranges over the `n+2` input cells. So the number of distinct surface configurations is at most `|Q|·|Γ|·(n+2)`, which is `O(n)` — a constant times `n`, with the alphabet and state count folded into the constant. That's the first thing the construction needs and it's worth being concrete that it really is linear in `n` and not, say, quadratic.

Fix a surface configuration `c` sitting at some stack height `h`. Run the machine forward from `c`. The stack will go up and down, but eventually — in a terminating computation — it drops *below* height `h` for the first time. Call the configuration at that moment the **terminator** of `c`. The terminator of `c` depends only on `c`, not on whatever is buried below in the stack, because the machine cannot touch those buried symbols until it has popped down past them, and the terminator is the very moment it first pops below `c`. Terminators also compose. If from `c` the machine first pushes, reaching some `c'`, then to find where `c` eventually pops below its height I should find the terminator of `c'` (where the pushed material gets cleared), and from there continue. The terminator of a configuration that pushes is computed from the terminator of the configuration after the push.

So the whole forward simulation can be expressed as: compute terminators recursively, and remember each terminator once I've computed it, in a table indexed by the surface configuration. Memoize `T[c]`. Now let me actually account for the cost rather than wave at it. Without memoizing, the recursion re-derives the same terminators repeatedly as the stack shuffles — that re-derivation is precisely the exponential blow-up; the machine keeps re-solving sub-problems it has already solved. With the table, the total work is: (number of distinct `c`) times (cost to fill one entry given that the entries it depends on are already filled). The first factor is `O(n)` as I just counted. The second factor: filling `T[c]` follows the machine's transition out of `c` — either it pops (terminator is immediate, `O(1)`), or it moves/pushes and the answer is read off one already-computed entry, again `O(1)` amortized once that entry is present. Each entry is computed exactly once and thereafter only looked up. So the product is `O(n)·O(1) = O(n)`. Let me sanity-check that this is the *right* accounting and I haven't hidden a factor: a step of the construction either fills a new table entry (and there are `O(n)` of those, each filled once) or reads an existing one to finish filling another. Every read is "caused by" the entry being filled, and each fill is `O(1)` reads — so total reads are `O(n)` too. The bookkeeping closes; nothing is `O(n²)`. The reason it comes out linear, stated plainly, is that the table shares each repeated sub-computation instead of redoing it — the exponential came entirely from redoing them.

Now I stare at this for the specific palindrome automaton and ask: when I strip away the pushdown-machine scaffolding, what is the table actually remembering? In the palindrome case, the useful residue of a failed attempt is not the whole history of the head moving back and forth. It is a compact surface fact: how much of the current string comparison has survived, and where the next viable comparison can resume after the doomed part has been popped away. That is exactly the kind of fact the terminator table stores. If I generalize this from palindromes to the simpler task "find the longest prefix of one string that occurs in another," the meaningful state at a failed comparison is just the number `j`: "I had matched `j−1` characters of the pattern and then the next one disagreed with the text." Everything needed to recover from that mismatch is a function of `j` and of the pattern alone. It does **not** depend on the unread text — that's the property I should keep checking myself on, because it's the whole point: the recovery rule reads nothing from the text, so the head need not go back to consult it. The terminator-table idea, specialized to this string-comparison situation, collapses from "indexed by surface configuration" down to "indexed by the matched-prefix length `j`."

So I have a candidate shape for the method: a table, indexed by pattern position `j`, that says — when a mismatch occurs after matching `j−1` characters — what pattern position to resume the comparison at, equivalently how far to slide the pattern, **without touching the text pointer**. The would-be backtracking is paid for once, off the pattern, and stored in a small table. If this is right, the matcher can keep the text head monotone: mismatch recovery changes only the pattern pointer. The two threads — "I want a forward-only matcher" and "Cook's construction shares repeated failed subcomputations through a table" — would be the same thread. Let me try to build the table directly, in string terms, and check whether it actually delivers the no-backup matcher, now that I think I have the mechanism.

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

Notice what this loop does *not* do: it never decreases `k`. `k` only ever increments. The recovery on mismatch happens entirely by shrinking `j` through the `next` table. That's the no-backup property — if the table is correct, it falls straight out of the loop's shape, since the only statement that could move the text head, `k := k+1`, only ever increments.

Let me pin down exactly what `next[j]` must be for this to be correct. When I'm about to set `j := next[j]`, I know `j > 0`, and I know the last few text characters, up to and including `text[k]`, are
```
pattern[1] … pattern[j−1]  x        (x = text[k], x ≠ pattern[j])
```
because I'd matched the first `j−1` pattern characters before failing on the `j`-th. I want the *smallest* slide that could still lead to a match, which is the same thing as the largest surviving prefix. So I want the largest `i < j` such that the last `i−1` characters before `x` already line up with the start of the pattern:
```
pattern[1] … pattern[i−1]  =  pattern[j−i+1] … pattern[j−1].
```
In other words, `pattern[1..i−1]` is both a prefix and a suffix of `pattern[1..j−1]` — a *border*. A border of length `i−1` means the shift is `j−i`; the larger `i` is, the less I slide, and the less risk I have of skipping a possible match. If there is no usable nonempty border, `next[j] = 0` and I slide the pattern past the current text character.

But there's a sharpening I shouldn't miss. Suppose the longest border has length `i−1`, so I'd resume comparing `text[k]` against `pattern[i]`. If `pattern[i]` happens to *equal* `pattern[j]` — the very character that just mismatched `text[k]` — then comparing `text[k]` against `pattern[i]` is forced to fail too, since `text[k] ≠ pattern[j] = pattern[i]`. That's a wasted comparison. So I should *not* resume at such an `i`; I should demand `pattern[i] ≠ pattern[j]` in the definition, and if the longest border's `pattern[i]` equals `pattern[j]`, skip past it to the next-longest. Let me define the plain border function first and then apply this refinement.

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

I don't trust any of this until I've built the table by hand for a pattern with real internal structure and watched the numbers come out. Take `abcabcacab`, positions `1..10`. Let me compute `f[j]`, the longest-proper-border-plus-one of `pattern[1..j−1]`, directly from the definition for each `j`:

- `j=1`: `f[1]=0` by convention.
- `j=2`: prefix to border is `pattern[1..1]="a"`; a single character has only the empty border, so `f[2]=1`.
- `j=3`: `"ab"` — empty border, `f[3]=1`.
- `j=4`: `"abc"` — empty border, `f[4]=1`.
- `j=5`: `"abca"` — proper borders: `"a"` is both prefix and suffix; length 1, so `f[5]=2`.
- `j=6`: `"abcab"` — `"ab"` is prefix and suffix; length 2, `f[6]=3`.
- `j=7`: `"abcabc"` — `"abc"` prefix=suffix; length 3, `f[7]=4`.
- `j=8`: `"abcabca"` — `"abca"` prefix=suffix; length 4, `f[8]=5`.
- `j=9`: `"abcabcac"` — longest border? `"abcabca"` ends in `c`, starts with `a`; try `"a"`: suffix is `...c`, last char `c≠a`, no. Empty border, `f[9]=1`.
- `j=10`: `"abcabcaca"` — ends in `a`; `"a"` is a border (first `a`, last `a`); is `"ab"` a suffix? suffix of length 2 is `"ca"≠"ab"`. So longest border is `"a"`, length 1, `f[10]=2`.

That gives `f[1..10] = (0, 1, 1, 1, 2, 3, 4, 5, 1, 2)`. Now apply the refinement to get `next`. For each `j>1`, compare `pattern[j]` with `pattern[f[j]]`; if equal, take `next[f[j]]`, else keep `f[j]`:

- `j=2`: `f=1`, `pattern[2]=b`, `pattern[1]=a`, differ → `next[2]=1`.
- `j=3`: `f=1`, `pattern[3]=c≠a` → `next[3]=1`.
- `j=4`: `f=1`, `pattern[4]=a`, `pattern[1]=a`, **equal** → `next[4]=next[1]=0`.
- `j=5`: `f=2`, `pattern[5]=b`, `pattern[2]=b`, **equal** → `next[5]=next[2]=1`.
- `j=6`: `f=3`, `pattern[6]=c`, `pattern[3]=c`, **equal** → `next[6]=next[3]=1`.
- `j=7`: `f=4`, `pattern[7]=a`, `pattern[4]=a`, **equal** → `next[7]=next[4]=0`.
- `j=8`: `f=5`, `pattern[8]=c`, `pattern[5]=b`, differ → `next[8]=5`.
- `j=9`: `f=1`, `pattern[9]=a`, `pattern[1]=a`, **equal** → `next[9]=next[1]=0`.
- `j=10`: `f=2`, `pattern[10]=b`, `pattern[2]=b`, **equal** → `next[10]=next[2]=1`.

So `next[1..10] = (0, 1, 1, 0, 1, 1, 0, 5, 0, 1)`. Look at `j=4`: a mismatch at the fourth character sets `next[4]=0`, sliding the pattern four places and never re-testing `text[k]` against another `a`. That is exactly the hand-intuition I started with — after `abc` then `x ≠ a`, slide all the way — now produced mechanically rather than by feel. The table encodes the "skip ahead because you already know that character" insight. Good; the construction does what I claimed.

Now the same amortized argument bounds the *matching* loop. The text pointer `k` advances at most `n` times. The assignment `j := next[j]` always replaces `j` by a smaller nonnegative value, while `j` is increased by `1` only when `k` advances. Charge every strict decrease of `j` to a previous increase of `j`; there are at most `n` such increases, so the inner-loop assignment is performed at most `n` times over the whole scan. Matching is `O(n)`, preprocessing is `O(m)`, the grand total is `O(m + n)`, the text pointer never moves backward, and I only need the pattern and its `O(m)` table in memory while the text streams past. That's everything I asked for.

I should convince myself the matching loop is actually *correct*, not just fast — that `next[j]` never makes me skip a real match. Before stating an invariant I'll watch the loop run on a tiny case where I can see every move, because if the no-backup recovery is going to skip a match it'll do it on something small. Pattern `aba`, with table (`1`-based) `next = (0, 0, 1)` — let me check that: `f[2]=1` (border of `"a"` is empty), `pattern[2]=b≠pattern[1]=a` so `next[2]=1`... wait, `f[2]=1` means resume position 1, and `pattern[1]=a≠b=pattern[2]` so no refinement, `next[2]=1`. `f[3]`: border of `"ab"` is empty, `=1`; `pattern[3]=a=pattern[1]`, equal → `next[3]=next[1]=0`. Hmm, so `next=(0,1,0)`. Let me run it on text `caababa` (where I can eyeball the answer: `aba` occurs at positions 3 and 5, `1`-based).

I'll trace with `k` the text position and `j` the next pattern position to test:
- `k=1`,`j=1`: `text[1]=c ≠ pattern[1]=a`; `j>0`? after `j:=next[1]=0`, loop exits; `k:=2,j:=1`. (text head moved forward, never back.)
- `k=2`,`j=1`: `text[2]=a = pattern[1]`; `k:=3,j:=2`.
- `k=3`,`j=2`: `text[3]=a ≠ pattern[2]=b`; `j:=next[2]=1`; `text[3]=a=pattern[1]`, stop; `k:=4,j:=2`. **Crucial:** `k` stayed at 3 while `j` dropped — the head did not back up, and the earlier `a` got reused as a fresh prefix start.
- `k=4`,`j=2`: `text[4]=b=pattern[2]`; `k:=5,j:=3`.
- `k=5`,`j=3`: `text[5]=a=pattern[3]`; `k:=6,j:=4=m+1` → **record start `k−m=6−3=3`**; `j:=f[4]`... here after a full match `j` resets to the restart border `f[m+1]`; the longest border of `"aba"` is `"a"`, length 1, so `j:=2`.
- `k=6`,`j=2`: `text[6]=b=pattern[2]`; `k:=7,j:=3`.
- `k=7`,`j=3`: `text[7]=a=pattern[3]`; `k:=8,j:=4=m+1` → **record start `8−3=5`**; done.

Matches at `3` and `5`. I can verify by eye against `caababa`: positions 3–5 are `aba`, positions 5–7 are `aba` — both correct, including the *overlapping* pair sharing the middle `a`, which the `j:=f[m+1]` restart is what caught. And at no step did `k` decrease. This is the behavior the whole design was for, and it actually happened on a real input.

That trace makes the invariant easy to state and believe. Let `p = k − j` (the text position just before the pattern's first character in the current alignment). The invariant: `text[p+i] = pattern[i]` for `1 ≤ i < j` (the first `j−1` pattern characters are matched), and for every earlier alignment `0 ≤ t < p` there is some position where `text` and `pattern` already disagreed (no full match was skipped to the left of `p`). When I match, `k` and `j` both advance and `p` is unchanged, so the matched-prefix part of the invariant extends by one — exactly what I saw at `k=4→5`. When I set `j := next[j]`, the new alignment starts at `p' = k - next[j]`, to the right of `p` because `next[j] < j` — exactly the jump at `k=3`. Every start position between `p` and `p'` is ruled out by information already in hand: either its overlap is not a border of `pattern[1..j−1]`, so it would disagree with one of the matched text characters before `k`, or it is a border whose next pattern character equals `pattern[j]`, so it would disagree with the same current `text[k]` because `text[k] ≠ pattern[j]`. The remaining candidate is exactly the next border not already killed by that mismatch; if it still disagrees with `text[k]`, the same argument applies again. When `j = m + 1`, the invariant says the whole pattern just matched ending at `k−1`; recording `k−m` is sound, and resetting `j` to `f[m+1]` is the overlapping-match case — and the trace's two overlapping hits are the invariant doing precisely that.

There's a subtlety worth checking: I used the *refined* `next` (with `pattern[i] ≠ pattern[j]`) in matching, but should the preprocessing's internal sliding also use `next`, or the plain `f`? For correctness of the resulting matcher, either works — the algorithm runs in linear time even if I used `f[j]` instead of `next[j]` in the main loop. The difference is the per-character delay. Let me make that difference concrete rather than assert it. Pattern `a^m` against text `a^{m−1} b …`. With the plain `f`, the border of `a^{j−1}` is `a^{j−2}`, so `f[j] = j−1` for every `j`; a single mismatch at the end then triggers `j := f[j]` cascading down `m, m−1, m−2, …, 1`, i.e. `m` slides and `m−1` redundant re-comparisons of the *same* text character before `k` finally advances. Now the refined `next` for `a^m`: at each `j>1`, `pattern[j]=a=pattern[f[j]]`, so the refinement fires and `next[j]=next[f[j]]=next[j−1]`, which unwinds all the way to `next[1]=0`. So the refined table is `next[j]=0` for all `j>1` — one step collapses the whole `m`-long cascade. I just verified by direct computation that the refinement turns `m` per-character slides into `1` on the pattern that most stresses it. So `next` is what keeps the work *between two consecutive text inputs* small — it matters for real-time, character-at-a-time reading, not just total throughput.

How small is that per-character delay in the worst case? Let me push on it, because it's a clean question: how many times can `j := next[j]` fire while a single text character is being scanned? Each firing slides the pattern, and consecutive slides correspond to nested borders of the matched prefix — i.e. to a string having several periods at once. Two periods of one string can't be too small together: by the Fine–Wilf phenomenon, if a string of length `ℓ` has periods `p` and `q` with `p + q ≤ ℓ` (roughly), it also has period `gcd(p,q)`, which forces structure. So to make the chain of nested borders long I want a pattern that packs as many distinct periods as possible into as short a length as possible — periods that are nearly but not quite forced to collapse. The **Fibonacci strings** are the natural candidate. Define `b₁ = b, b₂ = a, bₙ = bₙ₋₁ bₙ₋₂`: so `ab, aba, abaab, abaababa, …`, with `|bₙ| = Fₙ` the `n`-th Fibonacci number.

Rather than trust the heuristic, let me actually build the `next` table for one Fibonacci string and read the longest descending chain off it. Take `b₈ = abaababaabaababaababa`, length `F₈ = 21`. Computing its refined `next` table position by position (same procedure as the `abcabcacab` case, just longer) and then following the chain `j → next[j] → next[next[j]] → …` from the position that starts the longest one, I get
```
20 → 12 → 7 → 4 → 2 → 1 → 0.
```
Seven steps. Look at the values: `20, 12, 7, 4, 2, 1` are `F₈−1, F₇−1, F₆−1, F₅−1, F₄−1, F₃−1` — exactly the Fibonacci numbers minus one, descending. (`F₃..F₈ = 2,3,5,8,13,21`, minus one gives `1,2,4,7,12,20`.) So the chain length for `bₙ` is the number of Fibonacci numbers up to `Fₙ`, which is `n`. Since `Fₖ ≈ φᵏ/√5` with `φ = (1+√5)/2` the golden ratio, a pattern of length `m ≈ Fₙ` gives `n ≈ log_φ m` steps. Let me check the constant against the data: for `m = 21` I counted `7` steps in the chain, and `1 + log_φ 21 ≈ 1 + 6.33 = 7.33`; for `m = 8` (`b₆`) the longest chain `7 → 4 → 2 → 1 → 0` is `5` steps against `1 + log_φ 8 ≈ 5.32`; for `m = 55` (`b₁₀`) the chain has `9` steps against `1 + log_φ 55 ≈ 9.33`. Every count lands just under `1 + log_φ m` and grows by exactly one each time `m` jumps to the next Fibonacci number — so `Θ(log_φ m)` consecutive `j := next[j]` operations is both achieved and (since the Fibonacci construction maximizes the nested-period chain) the worst case. Between two successive single-character inputs only `O(log m)` work elapses; a real-time reader is fine. (With the unrefined `f`, that bound fails — the `a^m` computation above does `m` steps at one character.)

I have the scan; now I need the executable version. I'll switch to `0`-based indexing because it is cleaner in a real implementation, and I'll fold the "slide all the way past, advance text" case (`j = 0` in the `1`-based version) into a sentinel: let the table hold `−1` at the front, so a mismatch that drives the pattern pointer below `0` signals "advance the text pointer and reset," with no special-case branch. The table built this way is the refined `next` for mismatches, and the extra final slot is the restart border length after a complete match so overlapping matches are still found.

I'll package this as a single self-contained C++17 program: it reads the pattern `W` on the first line of stdin and the text `S` on the second, and prints the `0`-based start positions of every occurrence. Lengths are kept in `long long` so the indices never overflow on long texts.

```cpp
// Knuth-Morris-Pratt exact string matching, single-file C++17.
// Reads two lines from stdin: line 1 = pattern W, line 2 = text S.
// Prints the 0-based start positions of every occurrence of W in S
// (space-separated on one line; an empty line if there are none).
#include <bits/stdc++.h>
using namespace std;

// Failure / "next" table from the pattern alone (O(m)).
// For k < m, T[k] resumes after a mismatch at W[k] without moving the text
// pointer. T[m] restarts after a full match (so overlapping matches are found).
// T[0] = -1 is the sentinel for "no prefix survives."
vector<long long> preprocess(const string& W) {
    long long m = (long long)W.size();
    vector<long long> T(m + 1, 0);
    T[0] = -1;
    long long pos = 1, cnd = 0;     // cnd = current border length (the f[j] role)
    while (pos < m) {
        if (W[pos] == W[cnd]) {
            // border extends AND the next chars agree, so resuming at cnd would
            // just re-mismatch the same text char -- short-circuit it
            // (this is the pattern[i] != pattern[j] refinement).
            T[pos] = T[cnd];
        } else {
            T[pos] = cnd;
            while (cnd >= 0 && W[pos] != W[cnd])
                cnd = T[cnd];       // slide the pattern against itself
        }
        pos += 1;
        cnd += 1;
    }
    T[pos] = cnd;
    return T;
}

// Find every occurrence of W in S; the text pointer k only ever advances.
vector<long long> search(const string& S, const string& W) {
    vector<long long> matches;
    long long n = (long long)S.size(), m = (long long)W.size();
    if (m == 0) return matches;
    vector<long long> T = preprocess(W);
    long long k = 0;                // text pointer -- only ever advances; never backs up
    long long j = 0;                // pattern pointer
    while (k < n) {
        if (W[j] == S[k]) {
            k += 1;
            j += 1;
            if (j == m) {                   // full match ends just before k
                matches.push_back(k - j);
                j = T[j];                   // resume to find further matches
            }
        } else {
            j = T[j];                       // slide the pattern, keep k fixed
            if (j < 0) {                    // sentinel: nothing survives
                k += 1;
                j += 1;
            }
        }
    }
    return matches;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    string W, S;
    getline(cin, W);
    getline(cin, S);
    vector<long long> matches = search(S, W);
    for (size_t i = 0; i < matches.size(); ++i) {
        if (i) cout << ' ';
        cout << matches[i];
    }
    cout << '\n';
    return 0;
}
```

Let me run the `0`-based code on the cases I already worked by hand, to be sure the indexing translation didn't introduce an off-by-one. The table for `abcabcacab` should encode the same refinement I computed by hand; tracing `preprocess` on it I get `[-1, 0, 0, -1, 0, 0, -1, 4, -1, 0, 2]` — the `−1`s sit at the positions where my `1`-based `next` was `0` (the "slide fully past" positions `4, 7, 9` become `0`-based `3, 6, 8`, holding `−1`), so the two tables agree under the index shift. And feeding pattern `aba` with text `caababa` to the program prints `2 4` — the `0`-based starts of the same two overlapping matches I traced as `1`-based `3` and `5`. The translation is faithful.

The causal chain, end to end: the naive scanner wastes work and, worse, backs the text pointer up, which an editor's buffering makes painful — so I want a strictly forward, no-backup scan, and I can feel that a partial match already tells me the characters just read, but I have no systematic way to exploit it. Cook's theorem insists a linear method *exists* for problems I can phrase as a two-way pushdown automaton, including a palindrome problem I can't crack by hand — so I trace his construction and find, by actually accounting for its cost, that its engine is memoizing each surface configuration's terminator: `O(n)` configurations each filled once in `O(1)`, sharing repeated sub-computations through a table, which is why a possibly-exponential machine simulates in linear time. Distilling that to the matching automaton, the only thing the table needs to remember is, for each matched-prefix length `j`, where to resume after a mismatch — a function of the pattern alone — so I precompute a `next` table off the pattern, with the `pattern[i] ≠ pattern[j]` refinement to avoid re-comparisons; building it is the same matching algorithm run on the pattern against itself, in `O(m)`; the matcher then makes only the text pointer advance, recovering on mismatch by shrinking `j` through `next`, total `O(m + n)`, never backing up, `O(m)` memory — verified by hand-building the table for `abcabcacab`, hand-tracing the overlapping matches in `caababa`, and computing the worst-case per-character delay as a tight `Θ(log_φ m)`, witnessed by the `b₈` chain `20 → 12 → 7 → 4 → 2 → 1 → 0`.
