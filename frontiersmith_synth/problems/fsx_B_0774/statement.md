# Realize the Border Set

## Problem

A word $W$ of length $n$ over an alphabet consisting of the first $K$ lowercase
letters has a **border** of length $b$ ($1 \le b < n$) if its length-$b$
prefix equals its length-$b$ suffix: $W[0{:}b] = W[n-b{:}n]$. The **border
set** of $W$ is the collection of all such $b$.

You are given a target list of $m$ desired border lengths, each with a
positive weight, together with a spurious-border penalty $\lambda$ and an
alphabet-saving bonus $\alpha$. Output **one** word $W$ of length exactly $n$,
using only the first $K$ lowercase letters, so that its actual border set
matches the targets as well as possible while using as few distinct letters
as you can get away with.

## Input (stdin)

```
n K lam alpha
m
b_1 w_1
...
b_m w_m
```
$1 \le b_i \le n-1$ (all distinct), $w_i \ge 1$. The $m$ target lines are
**not** given in sorted order.

## Output (stdout)

Exactly $n$ whitespace-separated integers $w_0,\dots,w_{n-1}$, each in
$[0, K)$ — the word $W$, letter $i$ being $w_i$ (letter index $0$ stands for
`'a'`, index $1$ for `'b'`, and so on up to $K-1$). No other tokens.

## Feasibility

The output must contain exactly $n$ tokens, each parsing as an integer in
$[0,K)$. Any violation (wrong token count, non-integer token, out-of-range
letter index, empty output) scores 0.

## Objective (maximize)

Let $\text{Actual}(W) = \{\, b \in [1,n-1] : W[0{:}b]=W[n-b{:}n] \,\}$
(comparing the two length-$b$ integer subsequences).

- $\text{matched} = \sum w_i$ over targets whose $b_i \in \text{Actual}(W)$
- $\text{spurious} = |\text{Actual}(W) \setminus \{b_1,\dots,b_m\}|$
- $\text{distinct} = $ number of distinct letter values actually used in $W$

$$F(W) = \text{matched} - \lambda \cdot \text{spurious} + \alpha \cdot \max(0, K - \text{distinct}) + 2$$

The checker reports $\text{Ratio} = \min(1,\ F(W) / (18 \cdot B))$ where $B$
is a fixed, positive baseline: the value of $F$ (floored at $1$) on the
checker's own target-blind reference word, a golden-ratio low-discrepancy
rotation ($w_i = \lfloor (\{(i{+}1)\varphi\}) \cdot K \rfloor$) that does not
try to satisfy any target.

## The subtlety

Border lengths are **not** independently choosable. Write $p_i = n - b_i$ for
the *period* of border $b_i$. Two facts constrain which borders can coexist:
(1) if $p$ is a period, every multiple $kp<n$ is automatically a period too;
(2) (Fine and Wilf) if periods $p_1,p_2$ satisfy $p_1+p_2-\gcd(p_1,p_2)\le n$,
then $\gcd(p_1,p_2)$ is **forced** to be a period as well. Committing to a set
of target periods therefore mechanically forces a larger set — hence extra,
unavoidable borders — than the targets alone. Naively stitching each target
prefix onto its suffix independently, in the order the targets are listed,
ignores this: it typically fails to realize some targets (a later stitch can
overwrite and destroy an earlier one) and leaks far more spurious borders
than the math actually requires. A stronger construction figures out exactly
which extra borders any valid realization is forced to have, and builds the
word so that only the targets and those unavoidable extras appear — while
spreading distinct letters everywhere else to keep spurious borders and
wasted alphabet down.

## Constraints

$8 \le n \le 60$, $3 \le K \le 12$, $2 \le m \le 6$, $1 \le b_i \le n-1$,
$1 \le w_i \le 5$, $1 \le \lambda \le 4$, $0 \le \alpha \le 3$.

## Example

`n=6 K=3 lam=2 alpha=1`, one target `(b=2, w=3)`. The checker's own reference
word `1 0 2 1 0 2` (golden-ratio rotation, ignoring the target) has border set
$\{3\}$ only: it misses the target and has one spurious border, using all 3
letters, giving raw $F = 0 - 2\cdot1 + 1\cdot0 + 2 = 0$, floored to
$B=\max(1,0)=1$ — the checker's internal baseline (ratio $0$). A word such as
`0 1 2 0 0 1` has border set $\{2\}$ exactly (`0 1` prefix equals the `0 1`
suffix; no other length matches) — matched $=3$, spurious $=0$, distinct
$=3$, $F = 3 - 0 + 1\cdot(3-3) + 2 = 5$, giving ratio $\min(1,
5/(18\cdot1)) \approx 0.28$ — beating the baseline by realizing exactly the
target and nothing else.
