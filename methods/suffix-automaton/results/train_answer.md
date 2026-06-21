We are handed a string $s$ that grows one character at a time, and after each character we want the number of distinct non-empty substrings of the current prefix, all of it in $O(n)$ over a fixed alphabet. The honest first move is to keep a set of all substrings: when the prefix has length $m$ its substrings are the $\binom{m+1}{2}$ pieces $s[i..j)$, and the size of the set is the answer. It is correct and fine on a tiny string, but the count of substrings is $\Theta(m^2)$, so just enumerating them across a string of length $n$ is $\Theta(n^3)$ work and $\Theta(n^2)$ stored characters — dead on arrival at $n = 10^6$. The set-of-strings framing is the enemy from the start: we are storing and re-deriving overlapping pieces of a string we already hold, and almost all the work is redundant because substrings nest inside one another. The natural fix — a trie of all suffixes, since every substring is a prefix of some suffix — has the right semantics (a root path spells exactly a substring) but can carry $\Theta(n^2)$ nodes, because a string whose suffixes share little prefix forces nearly every substring into its own node. Path-semantics alone is cheap to state and ruinous to store; we need to collapse nodes without losing correctness.

I propose building the suffix automaton, the minimal deterministic automaton recognizing the suffixes of $s$. The reframing is this: make a directed graph with a start state $t_0$ and character-labelled edges such that the strings spelled by non-empty paths from $t_0$ are exactly the substrings of $s$. Then counting distinct substrings becomes counting distinct non-empty paths in a DAG, a quantity linear in the size of the graph. The whole question collapses to making the graph small while keeping it correct as $s$ grows. The device that shrinks the trie is a merge rule: two substrings can share a single state precisely when they have the same future, and in a string the future of a substring $t$ — which characters may legally follow it, and where each continuation leads — is governed entirely by *where it occurs*. So define $\mathrm{endpos}(t)$, the set of indices in $s$ at which an occurrence of $t$ finishes. Two substrings with identical ending-position sets are followed by exactly the same things in exactly the same places and are therefore interchangeable; we keep one state per $\mathrm{endpos}$ class, plus $t_0$ for the empty string.

What makes this work — and bounds its size — is the structure of those classes. Take two non-empty substrings $u, w$ with $|u| \le |w|$. If their ending sets share even one position $p$, then both end at $p$, so the length-$|u|$ string ending at $p$ is a suffix of the length-$|w|$ one, i.e. $u$ is a suffix of $w$; and then every occurrence of $w$ carries $u$ at its tail ending at the same spot, so $\mathrm{endpos}(w) \subseteq \mathrm{endpos}(u)$. If they share no position, the sets are disjoint. Hence any two ending sets are nested or disjoint, never crossing — they are *laminar*, and form a tree by inclusion. This is the load-bearing fact. From it the shape of a single class follows: sort a class by decreasing length; any two of its strings have the shorter as a proper suffix of the longer, so the lengths are strictly decreasing, and in fact a class is exactly the suffixes of its longest string over a *contiguous* length interval $[\mathrm{minlen}(v), \mathrm{len}(v)]$ — any suffix $v'$ of the longest string with length in that interval occurs only as that suffix (else it would drag the class's shortest string to a non-suffix spot) and so joins the class. Therefore state $v$ owns exactly $\mathrm{len}(v) - \mathrm{minlen}(v) + 1$ distinct substrings.

That already hands us the count, but $\mathrm{minlen}$ is awkward to carry, so I express it through the tree with the suffix link. Define $\mathrm{link}(v)$ to be the state of the longest proper suffix of $\mathrm{longest}(v)$ that lies in a *different* ending-position class. Walking the suffixes of $\mathrm{longest}(v)$ from long to short, the first several stay in $v$'s class and then one drops out; the shortest string still in $v$ has length one more than the longest string of $\mathrm{link}(v)$, so

$$\mathrm{minlen}(v) = \mathrm{len}(\mathrm{link}(v)) + 1.$$

The per-state count $\mathrm{len}(v) - \mathrm{minlen}(v) + 1$ then collapses to $\mathrm{len}(v) - \mathrm{len}(\mathrm{link}(v))$, and summing over all states gives the total number of distinct non-empty substrings:

$$\#\{\text{distinct non-empty substrings}\} = \sum_{v \ne t_0}\big(\mathrm{len}(v) - \mathrm{len}(\mathrm{link}(v))\big).$$

This needs no graph traversal at all — two integers per state and one sweep — which beats the equivalent path-DP $d[v] = 1 + \sum_{v \to w} d[w]$ (reporting $d[t_0] - 1$) because it requires no recursion and no extra array, reading the same fact straight off the per-state length intervals. The links, meanwhile, form a tree rooted at $t_0$ that is precisely the inclusion tree of the ending sets: $\mathrm{link}(v)$'s ending set strictly contains $v$'s, and following links shortens the string each step until reaching $t_0$, so the whole structure is self-consistent.

The real work is maintaining the machine online. Appending $c$ to $s$ injects exactly the strings $x + c$ for every suffix $x$ of $s$ (including $c$ itself, where $x$ is empty), because every brand-new substring of $s + c$ ends at the freshly appended position. The whole prefix $s + c$ is the longest of these and has a never-before-seen ending set, so it needs a fresh state $cur$ with $\mathrm{len}(cur) = \mathrm{len}(last) + 1$, where $last$ is the state of $s$. The suffix links are exactly how we reach all suffixes of $s$ in one structure: starting from $last$ and walking links visits states whose longest strings are the successively shorter suffixes of $s$. So we walk up from $last$, and at each state $p$ with no outgoing $c$-edge we know $x + c$ never occurred before, so we add $p \xrightarrow{c} cur$ and continue. We stop in one of two ways. If we fall off the top ($p = -1$), then no suffix of $s$ was ever followed by $c$ — the character is new — so $c$ has the broadest possible ending set and $\mathrm{link}(cur) = t_0$. Otherwise we hit a state $p$ that already has $p \xrightarrow{c} q$: the string $x + c$ already occurred inside $s$, we must not add a parallel edge, and we stop — leaving only $\mathrm{link}(cur)$ to set, which must point at the state representing $x + c$ exactly.

Here the design hinges on whether the existing edge is *tight*. If $\mathrm{len}(q) = \mathrm{len}(p) + 1$, then $q$'s longest string *is* $x + c$, so $\mathrm{link}(cur) = q$ and we are done. The hard case is $\mathrm{len}(q) > \mathrm{len}(p) + 1$, a *loose* edge: $q$'s longest string $y$ is strictly longer than $x + c$, with $x + c$ a proper suffix of $y$. In $s$ alone $x + c$ and $y$ shared an ending set — that is why they shared $q$ — but appending $c$ gives $x + c$ a new ending position (the appended index, since $x$ is a suffix of $s$) that $y$ does not get (a longer string, $y$ is not a suffix of $s + c$, or it would have surfaced earlier on the walk). Their classes have split, so $q$ must become two states. We clone $q$: make $clone$ a copy of $q$ with the same outgoing transitions and the same link but $\mathrm{len}(clone) = \mathrm{len}(p) + 1$. The copy keeps the transitions because every string leaving through $q$ still leaves the same way — we are only relabelling which state owns the short strings. The clone takes the short half of $q$'s interval and $q$ keeps the long half, so $\mathrm{link}(q) = clone$; and since the longest-proper-suffix-in-another-class of $cur$ is now exactly $\mathrm{longest}(clone)$, we set $\mathrm{link}(cur) = clone$ too. The final repair: the $c$-edges that were spelling $x + c$ and its shorter suffixes — exactly the suffix chain from $p$ upward — should now land in $clone$, so we continue up the links from $p$ redirecting each $c$-edge that still points to $q$, stopping when it points elsewhere or we fall off; edges spelling longer strings into $q$ are left alone, since they still want the long half. Then $last = cur$.

This is genuinely linear on all three counts. Each character adds one $cur$ and at most one $clone$, so there are at most $2n - 1$ states (independently confirmed by the laminar tree: at most $n$ leaves, and suppressing single-child internal nodes leaves every internal node with degree $\ge 2$, so at most $2n - 1$ nodes). Transitions are $O(n)$: tight edges form a spanning tree of longest paths in the DAG, and each loose edge charges injectively to a distinct proper non-empty suffix of $s$, of which there are at most $n - 1$. Time is $O(n)$ over a fixed alphabet: the edge-adding walk creates one transition per non-stopping step against the $O(n)$ transition budget; cloning copies $O(1)$ outgoing entries per clone over $O(n)$ clones; and the redirect walk is amortized by the standard monotone argument — the suffix position that becomes the second link ancestor after a redirect can only move right, at most $n$ times overall. The count rides along for free: a clone only partitions $q$'s old length interval, so $q$ plus $clone$ contributes exactly what $q$ did, and the only net increase per character is $\mathrm{len}(cur) - \mathrm{len}(\mathrm{link}(cur))$, which we add to a running total. Parallel arrays $\mathrm{len}$ and $\mathrm{link}$, a per-state dictionary of transitions, and that total are all the state we keep.

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
