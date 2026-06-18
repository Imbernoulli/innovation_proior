# Berlekamp–Massey (shortest linear recurrence) + polynomial-mod-power evaluation

## Problem

Given the first $2k$ terms of a sequence over $\mathbb{Z}_p$ ($p$ prime) that is
known to satisfy *some* linear recurrence of order $\le k$, recover the shortest
recurrence $c_0, \dots, c_{m-1}$ with
$$a_i = \sum_{j=0}^{m-1} c_j\, a_{i-j-1} \qquad (i \ge m),$$
then compute the $N$-th term ($N$ up to $10^{18}$) modulo $p$.

## Key idea

**Berlekamp–Massey: build the recurrence incrementally.** Scan the terms left to
right, maintaining the current shortest recurrence $C$ that fits everything seen so
far. At term $i$, predict $\hat a_i = \sum_j c_j a_{i-j-1}$ and form the
**discrepancy** $\delta_i = \hat a_i - a_i$.

- If $\delta_i = 0$, $C$ still fits — continue.
- The first time a nonzero term appears, no recurrence on all-zero history can
  produce it. Set $C$ to $i+1$ zeros, making that term part of the seed window,
  and record the empty recurrence's failed prediction as the reference failure.
- Otherwise repair $C$ with a **scaled, shifted residual from the last recurrence that
  failed.** Let that reference $B$ have failed at position $f$ with discrepancy
  $\mathrm{ld}$. The residual functional $a_t - \sum_j b_j a_{t-j-1}$ is $0$ on $B$'s
  correct range and $-\mathrm{ld}$ at $f$; the vector evaluating it is
  $(1, -b_0, -b_1, \dots)$. Move it from $f$ to $i$ by putting $i-f-1$ zero
  lookback slots before its former self-coefficient, so $+1$ lands at index $i-f-1$,
  and scale by $\kappa = \delta_i / \mathrm{ld}$. The result is silent at every
  earlier position and equals $a_i - \hat a_i$ at $i$, so $C + (\text{this})$
  predicts $a_i$ correctly and leaves the earlier terms intact:
  $$C \leftarrow C + \kappa(0,\dots,0,1,-b_0,-b_1,\dots),
  \qquad \kappa = \delta_i\,\mathrm{ld}^{-1}.$$
  The zero run has length $i-f-1$.
  Division by $\mathrm{ld}$ is why $p$ must be prime (Fermat inverse).

**Order grows only when forced.** The shifted fix has length $i-f+\lvert B\rvert$;
the order increases only if that exceeds $\lvert C\rvert$. Keep, as the reference
$B$ for the next failure, whichever past failure yields the shortest grown order —
switch to the just-overwritten $C$ exactly when
$i - f + \lvert B\rvert \ge \lvert C\rvert$. Each term costs $O(m)$, giving
$O(k^2)$ overall; the order is discovered, never guessed. With the true shortest
order $m \le k$, the supplied prefix includes at least the first $2m$ terms, enough
for the minimal recurrence to be forced by its own prefix.

**Evaluating $a_N$ for huge $N$ by polynomial mod-power.** Let
$f(x) = x^m - \sum_{j=0}^{m-1} c_j x^{m-1-j}$ be the characteristic polynomial and
$\Lambda(\sum_t g_t x^t) = \sum_t g_t a_t$. The recurrence says
$\Lambda(x^t f) = 0$ for all $t \ge 0$, so $\Lambda$ kills every multiple of $f$ and
$\Lambda(g)$ depends only on $g \bmod f$. Hence
$$a_N = \Lambda(x^N) = \Lambda(x^N \bmod f) = \sum_{i=0}^{m-1} (x^N \bmod f)_i\, a_i.$$
Compute $x^N \bmod f$ by binary exponentiation in $\mathbb{Z}_p[x]/(f)$:
square-and-multiply on the bits of $N$, each step a polynomial product (degree
$<2m$) reduced back to degree $<m$ via $x^e \equiv \sum_j c_j x^{e-1-j}$ swept from
the top. This is $O(m^2 \log N)$. (A companion-matrix exponentiation gives the same
answer in $O(m^3 \log N)$.)

## Algorithm

1. Scan the terms; maintain $C$, the last failing recurrence $B$, its position $f$,
   and its discrepancy $\mathrm{ld}$. Repair on each nonzero discrepancy as above,
   updating $B$ by the grown-order test.
2. Form $f(x)$ from $C$; compute $x^N \bmod f$ by binary exponentiation with
   multiply-then-reduce; return $\sum_i (x^N\bmod f)_i\,a_i$.

## Code

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
    cur = []            # current shortest recurrence so far
    last = []           # the recurrence current right before the last failure
    lf = -1             # failing index of `last`
    ld = 0              # discrepancy (predicted - actual) of `last` at lf
    for i in range(len(seq)):
        # predict a_i from the current recurrence
        t = 0
        for j in range(len(cur)):
            t = (t + seq[i - j - 1] * cur[j]) % p
        d = (t - seq[i]) % p          # discrepancy, predicted - actual
        if d == 0:
            continue                  # current recurrence still fits
        if not cur:
            # first nonzero term: set the order floor, record this failure
            cur = [0] * (i + 1)
            lf = i
            ld = d
            continue
        # build the scaled, shifted residual of the last failing recurrence
        k = d * inv(ld, p) % p        # discrepancy now / discrepancy then
        c = [0] * (i - lf - 1)
        c.append(k)                   # the +1-at-index-(i-lf-1), scaled
        for j in range(len(last)):
            c.append((-last[j] * k) % p)
        if len(c) < len(cur):
            c.extend([0] * (len(cur) - len(c)))
        prev = cur                    # remember pre-repair cur as a candidate ref
        c = [(c[j] + (cur[j] if j < len(cur) else 0)) % p
             for j in range(len(c))]
        # keep, as `last`, whichever reference grows the order least next time
        if i - lf + len(last) >= len(cur):
            last = prev
            lf = i
            ld = d
        cur = c
    return [x % p for x in cur]


def kth_term(rec, seq, N, p):
    """a_N mod p from rec = C[0..m-1] and the seed terms seq."""
    m = len(rec)
    if N < len(seq):
        return seq[N] % p
    if m == 0:
        return 0

    def mulmod(a, b):
        # multiply two deg<m polys, reduce mod x^m - sum rec[j] x^{m-1-j}
        r = [0] * (2 * m)
        for i in range(m):
            if a[i]:
                ai = a[i]
                for j in range(m):
                    r[i + j] = (r[i + j] + ai * b[j]) % p
        for e in range(2 * m - 1, m - 1, -1):
            if r[e]:
                re = r[e]
                for j in range(m):
                    r[e - 1 - j] = (r[e - 1 - j] + re * rec[j]) % p
        return r[:m]

    s = [0] * m
    s[0] = 1                          # accumulator = 1
    t = [0] * m
    if m == 1:
        t[0] = rec[0] % p             # x reduces to the constant c_0
    else:
        t[1] = 1                      # running square = x
    K = N
    while K:
        if K & 1:
            s = mulmod(s, t)
        t = mulmod(t, t)
        K >>= 1
    return sum(s[i] * seq[i] for i in range(m)) % p


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

## Complexity

- **`find_recurrence`:** $O(k^2)$ time ($O(k)$ terms, $O(m)$ predict + $O(m)$ repair
  each, with $m \le k$), $O(k)$ memory. The order $m$ is recovered, not assumed.
- **`kth_term`:** $O(m^2 \log N)$ time, $O(m)$ memory — $O(\log N)$ squarings, each a
  multiply-then-reduce in $O(m^2)$.
- Requires a prime modulus (the repair and the inverse both divide).
