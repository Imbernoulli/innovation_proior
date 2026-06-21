# Context

## Problem

Given a string `s` of length `n`, compute the Z-array, where `z[i]` is the length of the longest substring starting at `i` that is also a prefix of `s`; and use it to find all occurrences of a pattern `P` in a text `T`, all in `O(n)`.

By convention, `z[0]` is `0`. For example, `z_function("aaaaa")` should return `[0, 4, 3, 2, 1]`, `z_function("aaabaab")` should return `[0, 2, 1, 0, 2, 1, 0]`, and `z_function("abacaba")` should return `[0, 0, 1, 0, 3, 0, 1]`.

The direct approach computes each `z[i]` independently by comparing `s[0]` with `s[i]`, then `s[1]` with `s[i + 1]`, and so on until a mismatch or the end of the string. The goal is a linear-time `z_function`, and then a linear-time occurrence finder whose total work is proportional to `len(pattern) + len(text)`.

## Code framework

The lower-level routine returns the Z-array. The search routine returns the sorted list of start indices in `text` where `pattern` occurs, including overlapping occurrences.

```python
def z_function(s):
    """Return the Z-array of s: z[i] is the length of the longest prefix of s
    that also starts at position i. z[0] is 0 by convention."""
    n = len(s)
    z = [0] * n
    # TODO: fill in the linear-time computation.
    return z


def find_occurrences(text, pattern):
    """Return the sorted list of start indices i in text where pattern occurs."""
    m = len(pattern)
    if m == 0:
        return list(range(len(text) + 1))
    occurrences = []
    # TODO: fill in the linear-time search.
    return occurrences
```
