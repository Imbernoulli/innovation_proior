# Context: linear-time exact string matching

## Research question

Given a nonempty *pattern* string of length `m` and a long *text* string of length `n`, find every position in the text at which the pattern occurs as a contiguous substring (or just the leftmost). Text-editing programs do this constantly. The naive scanner that re-tries the pattern at every starting position costs `O(nm)` in the worst case, and — worse for a real editor reading the text from a file or buffer — it **backs up the input text**: after a partial match fails, it must reread characters it already passed. Backing up forces the editor to keep characters around in a buffer and complicates the buffering logic badly.

What a solution must achieve: scan the text **left to right without ever moving the text pointer backward**, find all matches in time proportional to `m + n` (with a small constant, so it is genuinely fast in practice, not just asymptotically), use memory proportional only to the pattern (`O(m)`) when the text streams in from a file, and stay independent of the alphabet size. A method that never backs up is exactly what removes the buffering pain and what makes a real-time, character-at-a-time reader possible.

## Background

The naive algorithm: align the pattern at text position `p`, compare `pattern[1], pattern[2], …` against `text[p+1], text[p+2], …`; on the first mismatch, advance `p` by one and start the comparison over from `pattern[1]`. On a pattern like `aaaaaaab` against `aaaa…aab` this does `≈ (n+1)·m`-ish work and repeatedly re-examines the same text characters. The structural waste: when the comparison fails after matching, say, `abc` and then hitting `x ≠ a`, the algorithm has just learned that the last four text characters were `a b c x`; it throws that knowledge away, backs the text pointer up, and rediscovers it.

Two bodies of prior theory bear directly on this.

**Finite automata for scanning.** A deterministic finite automaton reads its input strictly left to right, one character at a time, never backing up, in time linear in the input. Anything you can phrase as "recognize a regular set" you can scan without backup. Ken Thompson (1968) compiled regular expressions into code that searches text this way. The catch is that the pattern-matching done by an editor is most naturally written *with* backup (the naive loop above), and it is not obvious how to turn the "try-and-backtrack" formulation into a no-backup automaton, nor how cheaply.

**Cook's theorem on two-way deterministic pushdown automata (S. A. Cook, 1972).** A two-way deterministic pushdown automaton (2DPDA) is a finite-state control with a stack and a single read-only head that may move *both* directions on the input. Such a machine can run for an exponential number of steps before halting — forming, e.g., all stacks of `n` symbols takes `O(2^n)` steps. Cook proved the surprising result that **any language a 2DPDA recognizes, in any amount of running time, can be recognized on a random-access machine in `O(n)` time**. A *surface configuration* `c = (p, i, A)` records only the control state `p`, the head position `i`, and the top stack symbol `A`. In a terminating run, every configuration `c` has a unique *terminator* — the first later configuration whose stack has dropped below `c`'s current height. Terminators can be computed by recursion on the stack, and crucially **shared**: if you memoize each configuration's terminator in a table `T[c]`, no configuration is ever recomputed, and the whole exponential-time machine is simulated in time linear in the number of distinct configurations, which is `O(n)`. The sharing of terminators across the computation is precisely *why* the simulation collapses from exponential to linear.

Cook had shown in his Berkeley lectures that "even-palindromes-starred" is recognizable in `O(n log n)`; D. Chester had shown that the set of strings *beginning with* an even palindrome is recognizable by a 2DPDA, so by Cook's theorem it is RAM-recognizable in `O(n)` — even though no one could see how to do it by hand in less than about `n²` steps. That gap — a theorem guaranteeing a linear method exists for a problem one cannot solve linearly by inspection — is the live tension of the moment.

**Periodicity of strings.** A string `α` has *period* `p` if `α[i] = α[i+p]` for all valid `i`; equivalently `α = (α₁α₂)^k α₁`. Borders (a proper prefix that is also a suffix) and periods are two views of the same structure: `α` of length `ℓ` has a border of length `b` iff it has period `ℓ − b`. The Fine–Wilf theorem (1965) constrains how two periods of one string interact. These facts will govern exactly how far a pattern can be safely slid after a partial match, and how a pathological pattern can force many successive slides.

## Baselines

**Naive (brute-force) matching.** Align at every text start `p = 0, 1, 2, …`; compare forward until mismatch or full match; on mismatch reset to `pattern[1]` and increment `p`. Core idea: exhaustive re-trial. Cost: `O(nm)` worst case (`a^k b` in `a^N b`). Gap: re-examines text characters, and **backs up the text pointer**, so it needs the text kept available — the buffering problem in an editor. It discards the information that a partial match gives about the characters just read.

**Thompson's regular-expression search (1968).** Compile a regular expression into a nondeterministic automaton and simulate all active states at once, scanning the text once without backup; recognizing a regular expression of length `m` over a text of length `n` costs `O(mn)`. Core idea: automaton simulation, no backtracking on the text. Gap for the single-pattern problem: the `O(mn)` factor is exactly what we want to remove for the special, very common case where the "regular expression" is just a fixed literal string; the construction does no precomputation that exploits the pattern's internal repetitions.

**Cook's `O(n)` 2DPDA simulation (1972), used as a tool.** Not a string-matching algorithm per se, but a *constructive guarantee*: phrase the matching problem as a 2DPDA and you are handed a linear-time RAM procedure by the simulation. Core idea: memoized surface-configuration terminators. Gap: applied as a black box it produces a correct linear procedure but an opaque one, with a large constant and no insight into *which* characters get compared — one would have to "distill" the mechanism to get a practical algorithm.

## Evaluation settings

The natural yardsticks: worst-case running time as a function of `m` and `n` (the brute-force `O(nm)` is the bar to beat, the target is `O(m+n)`); the number of character comparisons; internal memory used when the text is read from an external file (target `O(m)`); the maximum delay between consecutive single-character inputs for a real-time reader; and dependence on alphabet size `q` (a good method should be independent of `q`). Pathological inputs to stress the worst case include highly self-similar patterns and texts such as `a^k b` in `a^N b`, and patterns with rich internal periodicity.

## Code framework

A streaming, no-backup scanner. The text arrives left to right; we keep a text pointer that only ever advances, a pattern pointer, and an empty preprocessing slot for whatever pattern-derived state is needed to choose the next comparison after a mismatch.

```python
def preprocess(pattern):
    # TODO: derive pattern-only state for resuming after a mismatch.
    pass

def search(text, pattern):
    table = preprocess(pattern)
    matches = []
    k = 0   # text pointer
    j = 0   # pattern pointer
    while k < len(text):
        # TODO: compare text[k] with pattern[j], update j from table on
        # mismatch, and record starts of full matches.
        pass
    return matches
```
