# Context

## Problem

Given a string `s` of length `n`, return a longest substring that reads the same forwards and backwards. The implementation should run in `O(n)` time and expose reusable center data for callers that need more than the single longest substring.

## Code framework

The input is a Python string. The lower-level routine returns per-center data; the public convenience routine uses that data to recover one longest palindromic substring.

```python
def manacher(s):
    """Return per-center data sufficient to recover palindromic substrings."""
    if not s:
        return [0]
    p = []
    # TODO
    return p


def longest_palindrome(s):
    """Return a longest palindromic substring of s in O(n) time."""
    if not s:
        return ""
    p = manacher(s)
    best_len, best_center = 0, 0
    # TODO
    start = 0
    # TODO
    return s[start:start + best_len]
```
