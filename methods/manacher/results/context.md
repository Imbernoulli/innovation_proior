# Context

## Problem

Given a string `s` of length `n`, return a longest substring that reads the same forwards and backwards.

## Code framework

The input is a Python string. A lower-level routine computes auxiliary data; a public convenience routine uses that data to return one longest palindromic substring.

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
