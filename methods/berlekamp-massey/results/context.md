# Context

## Problem

You are given the first $2k$ terms of a sequence over the field $\mathbb{Z}_p$ ($p$
prime). It is known to satisfy some linear recurrence of order $\le k$. Recover the
shortest such recurrence, then compute the $N$-th term ($N$ up to $10^{18}$) modulo
$p$.

A linear recurrence of order $m$ is a coefficient list $c_0, \dots, c_{m-1}$ such
that every term is the same fixed combination of the $m$ before it:
$$a_i = \sum_{j=0}^{m-1} c_j \, a_{i-j-1} \qquad \text{for all } i \ge m.$$
"Shortest" means smallest order $m$. The order is unknown in advance; only the
bound $m \le k$ is given, and the $2k$ supplied terms are exactly enough to pin a
recurrence of order $\le k$ down uniquely. Because $N$ can be as large as $10^{18}$,
generating the sequence term by term up to index $N$ is out of reach — once the
recurrence is known, the $N$-th term has to be obtained without walking through all
the intervening terms.

All arithmetic is modulo the prime $p$; division is multiplication by a modular
inverse.

## Code framework

The input is read as whitespace-separated integers: $p$, then $N$, then the count
of supplied terms, then the terms themselves. Modular arithmetic helpers (modular
power and modular inverse via Fermat's little theorem) are already available. What
is missing is the two pieces that do the work: `find_recurrence`, which turns the
supplied terms into the shortest recurrence, and `kth_term`, which evaluates the
$N$-th term from that recurrence.

```python
import sys


def power(a, b, p):
    a %= p
    r = 1
    while b:
        if b & 1:
            r = r * a % p
        a = a * a % p
        b >>= 1
    return r


def inv(a, p):
    return power(a, p - 2, p)  # Fermat; p prime


def find_recurrence(seq, p):
    """Shortest C with a_i = sum_j C[j]*a[i-j-1], over Z_p."""
    # TODO
    pass


def kth_term(rec, seq, N, p):
    """a_N mod p from rec = C[0..m-1] and the seed terms seq."""
    # TODO
    pass


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    p = int(next(it))
    N = int(next(it))
    cnt = int(next(it))
    seq = [int(next(it)) % p for _ in range(cnt)]
    rec = find_recurrence(seq, p)
    print(kth_term(rec, seq, N, p))


if __name__ == "__main__":
    main()
```
