# Suffix Automaton (SAM) — online distinct-substring count

## Problem

Given a string $s$ of length $n$, count the number of distinct non-empty substrings of $s$, with $s$ built one character at a time (online), in $O(n)$ over a fixed alphabet.

## Key idea

Build the **suffix automaton**: the minimal deterministic automaton for the suffixes of $s$. Every substring of $s$ is a prefix of some suffix, so it is exactly a path label starting from the initial state $t_0$. Distinct substrings then equal distinct non-empty paths from $t_0$.

**Endpos equivalence (the merge rule).** For a substring $t$, let $\mathrm{endpos}(t)$ be its set of ending positions in $s$. Two substrings share one state iff they have the same $\mathrm{endpos}$ — they are then followed by the same continuations everywhere, hence interchangeable. The automaton has one state per $\mathrm{endpos}$ class plus $t_0$.

**Structure of a class.** Endpos sets are *laminar*: for $|u|\le|w|$, either $u$ is a suffix of $w$ and $\mathrm{endpos}(w)\subseteq\mathrm{endpos}(u)$, or the sets are disjoint. Within one class the strings are exactly the suffixes of the longest string $\mathrm{longest}(v)$ over a contiguous length interval $[\mathrm{minlen}(v),\mathrm{len}(v)]$. So state $v$ owns $\mathrm{len}(v)-\mathrm{minlen}(v)+1$ distinct substrings.

**Suffix link.** $\mathrm{link}(v)$ is the state of the longest proper suffix of $\mathrm{longest}(v)$ lying in a *different* endpos class. Then $\mathrm{minlen}(v)=\mathrm{len}(\mathrm{link}(v))+1$, and the links form a tree rooted at $t_0$ (the inclusion tree of endpos sets).

**Distinct-substring count.** Summing the per-state counts and substituting $\mathrm{minlen}(v)=\mathrm{len}(\mathrm{link}(v))+1$:

$$\#\{\text{distinct non-empty substrings}\} \;=\; \sum_{v \ne t_0}\big(\mathrm{len}(v)-\mathrm{len}(\mathrm{link}(v))\big).$$

One sweep over the states gives the final count. Online, maintain the same sum incrementally: after `extend(c)`, the net new contribution is $\mathrm{len}(cur)-\mathrm{len}(\mathrm{link}(cur))$. If a clone is created, it only partitions $q$'s old length interval, so $q$ plus `clone` contributes exactly what $q$ contributed before.

## Online construction — `extend(c)`

Appending $c$ to $s$ injects exactly the strings $x+c$ for every suffix $x$ of $s$ (including $c$ itself). Let $last$ be the state of the whole string $s$.

1. Create $cur$ with $\mathrm{len}(cur)=\mathrm{len}(last)+1$ (the new whole string $s+c$).
2. Walk suffix links from $last$. While the current state $p$ has no $c$-transition, add $p\xrightarrow{c}cur$ and follow $\mathrm{link}(p)$. Stop at the first $p$ that already has a $c$-edge, or when falling off the top ($p=-1$).
3. If $p=-1$: $c$ is new, set $\mathrm{link}(cur)=t_0$.
4. Else let $q$ be the target of $p\xrightarrow{c}q$.
   - **Tight edge** $\mathrm{len}(p)+1=\mathrm{len}(q)$: $q$'s longest string is exactly $x+c$. Set $\mathrm{link}(cur)=q$.
   - **Loose edge** $\mathrm{len}(p)+1<\mathrm{len}(q)$: $x+c$ gains the new ending position but $q$'s longer strings do not, so $q$ splits. **Clone** $q$ into $clone$ — copy its transitions and its link, set $\mathrm{len}(clone)=\mathrm{len}(p)+1$. Walking up from $p$, redirect every $c$-edge that pointed to $q$ to $clone$. Set $\mathrm{link}(q)=\mathrm{link}(cur)=clone$.
5. Add $\mathrm{len}(cur)-\mathrm{len}(\mathrm{link}(cur))$ to the running answer, then set $last=cur$.

## $O(n)$ size and time

- **States** $\le 2n-1$ ($n\ge2$): each character adds one $cur$ and at most one $clone$. Independently, the endpos inclusion tree has at most $n$ leaves; after suppressing single-child internal nodes, every internal node has degree at least two, so the tree has at most $2n-1$ nodes.
- **Transitions** $O(n)$: tight edges form a spanning tree of longest paths, and loose edges can be charged injectively to proper suffixes.
- **Time** $O(n)$ over a fixed alphabet. The edge-adding walk creates one transition per non-stopping step; cloning copies $O(1)$ outgoing entries per clone; the redirect walk is amortized by the standard monotone suffix-position argument. With fixed transition arrays this is worst-case $O(n)$ time and $O(nk)$ memory for alphabet size $k$; the dictionary version below is expected $O(n)$ for a fixed alphabet.

## Code

```python
class SubstringCounter:
    """Online suffix automaton over integer character codes.

    One state per ending-position equivalence class (plus the root t0 = state 0).
    State v stores len[v] (length of its longest string), link[v] (the suffix
    link: the state of the longest proper suffix of longest(v) lying in another
    class), and trans[v] (character-code -> state). The distinct non-empty
    substrings owned by v are exactly the suffixes of longest(v) with lengths in
    [len[link[v]] + 1, len[v]], i.e. len[v] - len[link[v]] of them.
    """

    def __init__(self):
        self.length = [0]      # len of the root's longest string (empty) is 0
        self.link = [-1]       # root has no suffix link
        self.trans = [dict()]  # root's outgoing transitions
        self.last = 0          # state whose longest string is the whole prefix
        self.total = 0         # current number of distinct non-empty substrings

    def extend(self, c):
        # New state for the whole prefix s + c; its longest string grew by one.
        cur = len(self.length)
        self.length.append(self.length[self.last] + 1)
        self.link.append(-1)
        self.trans.append(dict())

        # Walk suffix links from last, attaching a c-edge to cur for every
        # suffix of s that was not yet followed by c (those x + c are new).
        p = self.last
        while p != -1 and c not in self.trans[p]:
            self.trans[p][c] = cur
            p = self.link[p]

        if p == -1:
            # c never followed any suffix of s: c is new, link cur to the root.
            self.link[cur] = 0
        else:
            q = self.trans[p][c]
            if self.length[p] + 1 == self.length[q]:
                # Tight edge: q's longest string is exactly x + c. Link straight.
                self.link[cur] = q
            else:
                # Loose edge: x + c gained the new ending position but q's longer
                # strings did not, so q's class splits. Clone q at length len(p)+1.
                clone = len(self.length)
                self.length.append(self.length[p] + 1)
                self.link.append(self.link[q])         # clone copies q's link
                self.trans.append(dict(self.trans[q]))  # and q's transitions
                # Redirect the suffix chain of c-edges from q over to the clone.
                while p != -1 and self.trans[p].get(c) == q:
                    self.trans[p][c] = clone
                    p = self.link[p]
                self.link[q] = clone
                self.link[cur] = clone

        self.last = cur
        self.total += self.length[cur] - self.length[self.link[cur]]
        return self

    def count_distinct_substrings(self):
        return self.total


def count_distinct_substrings(text):
    """Number of distinct non-empty substrings of text, built online."""
    sc = SubstringCounter()
    code = {}
    for ch in text:
        if ch not in code:
            code[ch] = len(code)
        sc.extend(code[ch])
    return sc.count_distinct_substrings()


def distinct_substrings_per_prefix(text):
    """For each prefix text[:i+1], the running count of distinct non-empty
    substrings -- one online pass."""
    sc = SubstringCounter()
    code = {}
    out = []
    for ch in text:
        if ch not in code:
            code[ch] = len(code)
        sc.extend(code[ch])
        out.append(sc.count_distinct_substrings())
    return out
```
