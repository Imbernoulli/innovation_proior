# Context

## Problem

Given a string $s$ of length $n$, count the number of distinct non-empty substrings of $s$, supporting the string being built one character at a time (online), in $O(n)$ over a fixed alphabet.

## Code framework

The string arrives online: a raw-text wrapper assigns stable dense integer codes as characters first appear, then feeds one code at a time. The driver below maintains some incremental state and exposes a query for the running answer; the per-character update and the query body are left to fill in.

```python
class SubstringCounter:
    """Online structure: feed one character code at a time with extend(c),
    then query count_distinct_substrings() for the number of distinct
    non-empty substrings of the prefix seen so far."""

    def __init__(self):
        # TODO: initialize the online state
        pass

    def extend(self, c):
        # TODO: incorporate the next character code c
        pass

    def count_distinct_substrings(self):
        # TODO: return the current count
        pass


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
