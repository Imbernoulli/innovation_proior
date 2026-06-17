# DistinctPalindromes

## Problem

Maintain the number of distinct non-empty palindromic substrings of a string while reading it online, one character at a time.

## Method

A length-$n$ string has at most $n$ distinct non-empty palindromic substrings. When a character is appended, every new palindrome must end at the new last position, and every such end-palindrome except the longest one already has an earlier mirrored occurrence inside the old prefix. Therefore one append creates at most one new distinct palindrome: the new longest palindromic suffix.

The structure stores one node per distinct palindrome, plus two roots:

- A length-$0$ root for the empty palindrome.
- A length-$-1$ imaginary root so wrapping a character around it creates a length-$1$ palindrome.
- A transition `go[x][c]` from an inner palindrome node `x` to the palindrome formed by wrapping character `c` on both ends.
- A suffix link `fail[x]` to the longest proper palindromic suffix of node `x`.

To append character `c`, walk suffix links from the current longest palindromic suffix `last` until the candidate palindrome can be wrapped by `c`, i.e. until `s[i - 1 - length[x]] == s[i]`. The landing node `cur` gives the new longest palindromic suffix through `go[cur][c]`. If that transition is missing, create exactly one new node and set its suffix link by repeating the same search from `fail[cur]`; a single-character node links to the length-$0$ root.

The first suffix-link search over all appends is amortized linear: if the walk stops at length $L$, the new longest palindromic suffix has length $L+2$, so the failed steps are at most $\Phi_{\text{old}}-\Phi_{\text{new}}+2$ for $\Phi=$ current longest-suffix length. The second search, used only for new nodes, is the same proper-suffix-chain search and is charged by the same length-drop potential, adding only a constant multiple of the suffix-link work. Thus the total number of suffix-link jumps is $O(n)$. With array transitions over a fixed alphabet, construction is $O(n)$ time; with balanced maps it is $O(n\log\Sigma)$ time and $O(n)$ sparse transition space.

## Code

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
