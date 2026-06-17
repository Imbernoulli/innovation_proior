In front of me is a string $s$, growing one character at a time, and I want to keep track of all of its *distinct* palindromic substrings: how many there are, and how that number changes the instant a new character lands. The first idea is correct but unusable. Every substring is some $s[l..r]$, there are $\Theta(n^2)$ of them, and I can test each one against its reverse and put the palindromic strings into a set. If I compare characters directly, the tests and stored strings push the work toward $O(n^3)$; if I add rolling hashes or palindrome radii, I can reduce the testing cost, but I am still treating substrings as a quadratic collection of candidates. The online version is even worse in spirit: each new character asks me to revisit a large suffix-ending family. I need to stop looking at all substrings and ask whether the *distinct palindromes themselves* are only a small family.

The example $a^n$ is useful because it is dense in palindromes but not quadratic in distinct ones. Its distinct non-empty palindromic substrings are $a, aa, aaa, \ldots, a^n$, exactly $n$. That suggests the real upper bound might be $n$, not $n^2$. If that is true, then a linear-size online object is plausible.

Let me prove the bound before I build anything on it. Suppose I have already processed $s'$ and append a final character $c$, forming $s=s'c$. The only palindromic substrings that could be new are the palindromes ending at this new last position. List their left endpoints as $l_1 < l_2 < \cdots < l_k$, so $s[l_1..|s|]$ is the longest end-palindrome and $s[l_k..|s|]$ is the shortest one. For any $i \ge 2$, the palindrome $s[l_i..|s|]$ is a suffix of the longer palindrome $s[l_1..|s|]$. Mirror symmetry inside the longer palindrome sends that suffix to the prefix $s[l_1..l_1+|s|-l_i]$ of the same length. Because $s[l_i..|s|]$ is itself a palindrome, reversing that mirrored prefix gives the same string, so the suffix string equals that prefix string. Its right endpoint is $l_1+|s|-l_i$, which is strictly less than $|s|$ because $l_i>l_1$. So every end-palindrome except possibly the longest one has already appeared wholly inside $s'$. Appending one character can create at most one new distinct palindrome, and if it creates one, it is the longest palindromic suffix of the new string. The one-character base case gives at most $n$ distinct non-empty palindromic substrings in a length-$n$ string, and $a^n$ shows the bound is tight.

That result gives me the update rule. I can store one node per distinct palindrome; each appended character creates zero or one node; the answer is the number of non-root palindrome nodes. The remaining problem is how to find the new longest palindromic suffix without scanning the whole prefix.

A palindrome has a natural smaller palindrome inside it. If $t$ has length at least $2$, then $t=cuc$ for some character $c$ and some inner palindrome $u$. That makes $u$ the parent of $t$, and the edge label is the character used to wrap both ends. From any node, there is at most one outgoing edge for a fixed character, because wrapping the same inner palindrome with the same character produces one string. Even-length palindromes bottom out at the empty string, so I need a root of length $0$. Odd-length palindromes bottom out at a single character, and a single character would have an inner object of length $-1$ if the same "strip two ends" rule were to keep working. I can introduce an imaginary root of length $-1$; wrapping a character around it gives a length-$1$ palindrome. With roots of lengths $0$ and $-1$, every non-empty palindrome is obtained by the same parent-plus-wrapping rule.

Now I need the search path for an append. Assume `last` points to the longest palindromic suffix of the current string. When I append a new character $c$ at position $i$, any new palindromic suffix longer than one character must look like $c\,t\,c$, where $t$ is a palindromic suffix of the old string and the character immediately before that occurrence of $t$ is also $c$. Among all such $t$, the longest one gives the new longest palindromic suffix. So I need to enumerate the current palindromic suffixes from longest to shortest.

That calls for a second pointer on each node: the longest proper palindromic suffix of that node's palindrome. I call it `fail`. Starting from `last` and repeatedly following `fail` visits the palindromic suffixes of the current string in decreasing length, because every palindromic suffix of the whole string is a palindromic suffix of its longest palindromic suffix. Both roots can fail to the length-$-1$ root, so the walk always has a terminal node.

The wrap test is local once I know a candidate length. After appending $c$, the new character is at index $i$ in the stored sequence. If node $x$ represents a palindromic suffix of the old string of length `length[x]`, that suffix occupies the old positions $i-\text{length}[x]$ through $i-1$. To wrap it with the new character, the character immediately before it, at $i-1-\text{length}[x]$, must equal the new character at $i$. So I walk `fail` until `s[i - 1 - length[x]] == s[i]`. For the length-$-1$ root this becomes `s[i] == s[i]`, always true, so a single-character palindrome is always available. For the length-$0$ root it becomes `s[i-1] == s[i]`, exactly the condition for a length-$2$ palindrome. A guard value before the first real character handles failed tests at the left boundary.

Once the walk lands at `cur`, the desired suffix is the palindrome obtained by wrapping `cur` with `c`. If the `c` transition from `cur` already exists, the palindrome was seen before; I reuse that node and the distinct count does not change. If the transition is absent, this is the one new palindrome allowed by the bound, so I create a node of length `length[cur] + 2`, add the transition, and increment the count.

The new node still needs its own `fail` pointer. Its proper palindromic suffixes come from taking proper palindromic suffixes of `cur` and wrapping the first one whose preceding character also matches $c$. That is the same test, but started from `fail[cur]`, not from `cur` itself. If that second walk lands at `v`, then the longest proper palindromic suffix of the new node is the existing `c` child of `v`. The single-character case must be special: its longest proper palindromic suffix is the empty palindrome, so its `fail` is the length-$0$ root. Without that special case, the general transition lookup would point back at the single-character node being created, which is not a proper suffix.

Then `last` becomes the node reached by the `c` transition from `cur`, because that node is exactly the new longest palindromic suffix. The answer for the current prefix is the node count excluding the two roots, or equivalently a running counter incremented only when a new node is created.

I still have to account for the walks. A single append can climb a long `fail` chain, so worst-case constant time per character is too strong. The useful statement is amortized. Let $\Phi_i$ be the length of `last` after processing the first $i$ characters, with $\Phi_0=0$. During the first walk for character $i$, each failed suffix-link step strictly decreases the candidate length. If the walk stops at a candidate of length $L$, the new `last` has length $L+2$, so the number of failed steps is at most $\Phi_{i-1}-L=\Phi_{i-1}-\Phi_i+2$. Summing over all appends gives at most $2n+\Phi_0-\Phi_n=O(n)$ failed steps for the first walks. The second walk is the same kind of search, but on the proper-suffix chain used to assign the new node's `fail`: it starts from `fail[cur]`, every failed step again spends a strict drop in candidate length, and the standard split of the two searches charges this proper-suffix contribution by the same potential argument, giving another constant multiple of the first contribution. With array transitions over a fixed alphabet, each transition check is constant time, so construction is $O(n)$ total with $O(n\Sigma)$ transition storage. With balanced maps, the suffix-link work is still linear and each transition lookup costs $O(\log \Sigma)$, giving $O(n\log\Sigma)$ time and $O(n)$ stored edges; with hash maps the expected transition factor is constant.

The node and edge counts are linear for the same reason the bound was linear. There are at most $n$ non-root palindrome nodes, plus the two roots. Every non-root node has one incoming wrapping edge from its inner palindrome with one label, so the number of stored transitions is also $O(n)$ when sparse maps are used.

Let me write the code around the public online counter. Each node stores its palindrome length, its `fail` pointer, and its transition dictionary. The processed characters are stored with a private guard at index zero, `last` tracks the current longest palindromic suffix, and `_count` is the number of distinct non-empty palindromes discovered so far.

```python
class DistinctPalindromes:
    """Online counter for distinct non-empty palindromic substrings."""

    ODD = 0
    EVEN = 1

    def __init__(self):
        self._guard = object()
        self.s = [self._guard]
        self.length = [-1, 0]
        self.fail = [self.ODD, self.ODD]
        self.go = [dict(), dict()]
        self.last = self.EVEN
        self._count = 0

    def _walk(self, x):
        i = len(self.s) - 1
        while self.s[i - 1 - self.length[x]] != self.s[i]:
            x = self.fail[x]
        return x

    def add(self, c):
        self.s.append(c)
        cur = self._walk(self.last)
        created = False

        if c not in self.go[cur]:
            now = len(self.length)
            self.length.append(self.length[cur] + 2)
            self.go.append(dict())

            if self.length[now] == 1:
                self.fail.append(self.EVEN)
            else:
                f = self._walk(self.fail[cur])
                self.fail.append(self.go[f][c])

            self.go[cur][c] = now
            self._count += 1
            created = True

        self.last = self.go[cur][c]
        return created

    def count_distinct(self):
        return self._count


def count_distinct_palindromes(text):
    """Number of distinct non-empty palindromic substrings of text, built online."""
    dp = DistinctPalindromes()
    code = {ch: i for i, ch in enumerate(sorted(set(text)))}
    for ch in text:
        dp.add(code[ch])
    return dp.count_distinct()


def distinct_palindromes_per_prefix(text):
    """For each prefix text[:i+1], the running count of distinct non-empty
    palindromic substrings -- one online pass."""
    dp = DistinctPalindromes()
    code = {ch: i for i, ch in enumerate(sorted(set(text)))}
    out = []
    for ch in text:
        dp.add(code[ch])
        out.append(dp.count_distinct())
    return out
```

The construction follows the chain forced by the problem: distinct palindromes are at most one per appended character because all shorter end-palindromes already have an earlier mirrored occurrence; that makes one node per palindrome viable; wrapping an inner palindrome by one character gives the parent-transition relation and requires roots of lengths $0$ and $-1$; suffix links enumerate the palindromic suffixes I must test; the wrap test finds the longest suffix that can be extended; and the length-drop potential pays for the suffix-link walks, so the online structure has linear total construction time over a fixed alphabet while maintaining the distinct-palindrome count after every prefix.
