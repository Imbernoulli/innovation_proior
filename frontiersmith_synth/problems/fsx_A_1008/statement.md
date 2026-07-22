# One Word to Reset Them All

## Problem

You are given a family of $k$ complete deterministic finite automata (DFAs)
that all share the same input alphabet $\Sigma = \{0, 1, \dots, \sigma-1\}$.
Automaton $i$ has states $\{0, \dots, n_i - 1\}$ and a total transition
function $\delta_i(s, l)$ defined for every state $s$ and every letter $l$.
The automata may share fragments of transition structure with each other —
this is **not** flagged in the input; noticing and exploiting it (if present)
is your job.

For a word $W = w_1 w_2 \cdots w_m$ over $\Sigma$, apply it to automaton $i$
**from every state at once**: start with $S_0 = \{0,\dots,n_i-1\}$ and set
$S_j = \{\, \delta_i(s, w_j) : s \in S_{j-1} \,\}$. Automaton $i$ is *fully
reset* by $W$ if the final set $S_m$ has exactly one element; more generally,
the amount of *collapse* $W$ achieves on automaton $i$ is how much $S_m$
shrank relative to $\{0,\dots,n_i-1\}$.

**Useful fact:** because every $\delta_i$ is a deterministic total function,
once $S_j$ has shrunk to a single element it stays a single element for every
later letter. A reset, once achieved, is never undone by appending more
letters — only other, not-yet-reset automata can still be affected by what
comes next.

## Input (stdin)

```
k sigma Lmax
n_0
<row for letter 0, n_0 integers>
...
<row for letter sigma-1, n_0 integers>
n_1
... (same, sigma rows of n_1 integers)
...
```

Row $l$ of automaton $i$'s block lists $\delta_i(s, l)$ for $s = 0,\dots,n_i-1$
in order. $L_{max}$ is a hard word-length budget shared by all automata.

## Output (stdout)

One line containing a single word $W$: a (possibly empty) string of digit
characters from `'0'` to the character for $\sigma-1$. $W$ is applied,
unmodified, to every one of the $k$ automata simultaneously as described
above.

## Feasibility

- $|W| \le L_{max}$.
- Every character of $W$ must be a valid letter (a digit in $[0, \sigma-1]$).
- The output must be a single line (or empty).

Any violation scores `Ratio: 0.0`.

## Objective / Scoring

For automaton $i$ let $m_i = |S_m|$ after applying the full word. Define
$\text{contribution}_i = (n_i - m_i) / (n_i - 1) \in [0,1]$ (1.0 = fully
reset, 0.0 = no collapse). Your raw score is $F = \sum_i \text{contribution}_i$.

The checker also computes its own weak internal baseline $B$ (repeating one
fixed letter, ignoring the transition tables) and reports

```
sc = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```

Bigger $F$ relative to $B$ is better; the cap leaves headroom above what a
merely competent strategy achieves.

## Constraints

$1 \le k \le 12$, $2 \le \sigma \le 6$, $2 \le n_i \le 16$, $L_{max}$ roughly
$25$–$50$ (given per test). Time limit 5s, each `.in` well under 5 MB.

## Example (illustrative shapes only — not a real test)

Two automata, $\sigma=2$, $L_{max}=5$:

- Automaton 0 ($n_0=3$): letter 0: `0 0 1`, letter 1: `1 2 2`.
- Automaton 1 ($n_1=3$): letter 0: `1 2 0`, letter 1: `0 1 0`.

With $W = \texttt{"11"}$: automaton 0 starts $\{0,1,2\}$, letter 1 gives
$\{1,2\}$, letter 1 again gives $\{2\}$ — fully reset,
contribution $= 1.0$. Automaton 1 starts $\{0,1,2\}$, letter 1 gives
$\{0,1,0\}=\{0,1\}$, letter 1 again gives $\{0,1\}$ (a fixed set from here) —
contribution $= (3-2)/(3-1) = 0.5$. So $F = 1.5$ for this (short, non-optimal)
example word; a longer or differently chosen word could do better on
automaton 1 too.
