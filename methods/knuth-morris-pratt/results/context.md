# Context: linear-time exact string matching

## Research question

Given a nonempty *pattern* string of length `m` and a long *text* string of length `n`, find every position in the text at which the pattern occurs as a contiguous substring (or just the leftmost). Text-editing programs perform this operation constantly. The naive scanner that re-tries the pattern at every starting position costs `O(nm)` in the worst case. How can one match a fixed literal pattern against a text in time proportional to `m + n`, using memory proportional only to the pattern, and independently of the alphabet size?

The deliverable is a single self-contained C++ program that reads from stdin and writes to stdout. It reads the pattern `W` from the first input line and the text `S` from the second input line, then prints the `0`-based start positions of every occurrence of `W` in `S` on one space-separated line, followed by a newline.

## Background

The naive algorithm: align the pattern at text position `p`, compare `pattern[1], pattern[2], …` against `text[p+1], text[p+2], …`; on the first mismatch, advance `p` by one and start the comparison over from `pattern[1]`. On a pattern like `aaaaaaab` against `aaaa…aab` this does `≈ (n+1)·m`-ish work and repeatedly re-examines the same text characters. After a partial match it backs the text pointer up, which forces the editor to keep characters around in a buffer.

Two bodies of prior theory bear directly on this.

**Finite automata for scanning.** A deterministic finite automaton reads its input strictly left to right, one character at a time, never backing up, in time linear in the input. Anything you can phrase as "recognize a regular set" you can scan without backup. Ken Thompson (1968) compiled regular expressions into code that searches text this way. The challenge is that the pattern-matching done by an editor is most naturally written *with* backup (the naive loop above), and it is not obvious how to turn the "try-and-backtrack" formulation into a no-backup automaton, nor how cheaply.

**Cook's theorem on two-way deterministic pushdown automata (S. A. Cook, 1972).** A two-way deterministic pushdown automaton (2DPDA) is a finite-state control with a stack and a single read-only head that may move *both* directions on the input. Such a machine can run for an exponential number of steps before halting — forming, e.g., all stacks of `n` symbols takes `O(2^n)` steps. Cook proved the surprising result that **any language a 2DPDA recognizes, in any amount of running time, can be recognized on a random-access machine in `O(n)` time**. The proof is *constructive*: it does not merely assert the linear procedure exists, it builds one out of the given pushdown machine.

Cook had shown in his Berkeley lectures that "even-palindromes-starred" is recognizable in `O(n log n)`; D. Chester had shown that the set of strings *beginning with* an even palindrome is recognizable by a 2DPDA, so by Cook's theorem it is RAM-recognizable in `O(n)`.

**Periodicity of strings.** A string `α` has *period* `p` if `α[i] = α[i+p]` for all valid `i`; equivalently `α = (α₁α₂)^k α₁`. Borders (a proper prefix that is also a suffix) and periods are two views of the same structure: `α` of length `ℓ` has a border of length `b` iff it has period `ℓ − b`. The Fine–Wilf theorem (1965) constrains how two periods of one string interact. These are standard facts about the internal repetition structure of strings.

## Baselines

**Naive (brute-force) matching.** Align at every text start `p = 0, 1, 2, …`; compare forward until mismatch or full match; on mismatch reset to `pattern[1]` and increment `p`. Core idea: exhaustive re-trial. Cost: `O(nm)` worst case (`a^k b` in `a^N b`).

**Thompson's regular-expression search (1968).** Compile a regular expression into a nondeterministic automaton and simulate all active states at once, scanning the text once without backup; recognizing a regular expression of length `m` over a text of length `n` costs `O(mn)`. Core idea: automaton simulation, no backtracking on the text.

**Cook's `O(n)` 2DPDA simulation (1972), used as a tool.** Not a string-matching algorithm per se, but a *constructive guarantee*: phrase the matching problem as a 2DPDA and the construction hands back a linear-time RAM procedure. Core idea: a constructive linear-time simulation of the pushdown machine.

## Evaluation settings

The natural yardsticks: worst-case running time as a function of `m` and `n` (the brute-force `O(nm)` is the bar to beat, the target is `O(m+n)`); the number of character comparisons; internal memory used when the text is read from an external file (target `O(m)`); the maximum delay between consecutive single-character inputs for a real-time reader; and dependence on alphabet size `q` (a good method should be independent of `q`). Pathological inputs to stress the worst case include highly self-similar patterns and texts such as `a^k b` in `a^N b`, and patterns with rich internal periodicity.

## Code framework

A single-file C++17 streaming scanner scaffold. The input parsing and output formatting are fixed; the algorithm body fills `matches` with the start positions to print.

```cpp
#include <bits/stdc++.h>
using namespace std;
int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    string W, S;
    getline(cin, W);
    getline(cin, S);
    vector<long long> matches;
    // TODO: fill matches with the 0-based start positions where W occurs in S.
    for (size_t i = 0; i < matches.size(); ++i) {
        if (i) cout << ' ';
        cout << matches[i];
    }
    cout << '\n';
    return 0;
}
```
