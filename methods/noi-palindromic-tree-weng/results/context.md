## Problem

Given a string of length $n$, maintain the number of distinct non-empty palindromic substrings while the string is read one character at a time. After each appended character, the current prefix's count should be available without rebuilding from scratch. The target is an online implementation whose total work is linear over a fixed alphabet if a compact representation can be found.

## Code framework

```python
class DistinctPalindromes:
    """Online counter: feed one character at a time with add(c), then query
    count_distinct() for the number of distinct non-empty palindromic substrings
    seen so far."""

    def __init__(self):
        # TODO: initialize the online state
        pass

    def _walk(self, x):
        # TODO: internal search helper
        pass

    def add(self, c):
        # TODO: incorporate the next character
        pass

    def count_distinct(self):
        # TODO: return the current count
        pass


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
