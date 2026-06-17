# Lifting the Exponent (LTE)

## Problem solved

Given a base difference $a^n - b^n$ (or sum $a^n + b^n$) and a prime $p$, determine the exact power of $p$ dividing it — its $p$-adic valuation $v_p$ — without factoring the (typically enormous) number. The lemma reduces this to data about the base and the exponent alone.

## Notation

For a prime $p$ and a nonzero integer $x$, $v_p(x)$ is the **$p$-adic valuation**: the largest integer $k$ with $p^k \mid x$. So $p^{v_p(x)} \,\|\, x$ (exactly divides). The largest power of $p$ dividing $x$ is $p^{v_p(x)}$.

## The lemma (all cases)

Let $a, b$ be nonzero integers and $n \ge 1$, with the displayed base difference or sum nonzero.

**Odd prime, difference.** Let $p$ be an odd prime with $p \mid a - b$ and $p \nmid a$, $p \nmid b$. Then for every $n \ge 1$,
$$v_p(a^n - b^n) = v_p(a - b) + v_p(n).$$

**Odd prime, sum.** Let $p$ be an odd prime with $p \mid a + b$ and $p \nmid a$, $p \nmid b$. Then for every **odd** $n$,
$$v_p(a^n + b^n) = v_p(a + b) + v_p(n).$$

**$p = 2$.** Let $a, b$ be **odd** integers.
- Difference, $n$ **odd**: $\quad v_2(a^n - b^n) = v_2(a - b)$.
- Difference, $n$ **even**:
$$v_2(a^n - b^n) = v_2(a - b) + v_2(a + b) + v_2(n) - 1.$$
  In particular, when $4 \mid a - b$ this reduces to $v_2(a^n - b^n) = v_2(a - b) + v_2(n)$ (since then $v_2(a+b)=1$), matching the odd-prime shape. The correction term $v_2(a+b)$ and the $-1$ are only nontrivial when $2 \,\|\, (a - b)$.
- Sum, $n$ **odd**: $\quad v_2(a^n + b^n) = v_2(a + b)$.
- Sum, $n$ **even**: $\quad v_2(a^n + b^n) = 1$.

## Why

**Odd prime.** From $a^n - b^n = (a - b)\sum_{i=0}^{n-1} a^{n-1-i} b^{i}$, the cofactor is $\equiv n b^{n-1} \pmod p$ when $a \equiv b \pmod p$, so for $p \nmid n$ all the $p$ sits in $a - b$ and $v_p = v_p(a-b)$. Replacing the exponent $m$ by $pm$ multiplies $a^m - b^m$ by
$$Q=\frac{a^{pm}-b^{pm}}{a^m-b^m}=\sum_{j=0}^{p-1} a^{m(p-1-j)} b^{mj}.$$
Set $D=a^m-b^m$. Since $a^m=b^m+D$ and $p\mid D$, expansion to first order in $D$ gives
$$Q\equiv p\,b^{m(p-1)}+\frac{p(p-1)}{2}b^{m(p-2)}D \pmod{D^2}.$$
The second term and the error are divisible by $p^2$, while $b^{m(p-1)}$ is a unit modulo $p$. Thus $Q$ is divisible by $p$ but not $p^2$. Each factor of $p$ in $n$ adds exactly $1$ to $v_p$, giving $v_p(a-b) + v_p(n)$ by induction. The sum form follows by applying the difference form to $a$ and $-b$ for odd $n$.

**$p = 2$.** The "$\times 2$ in the exponent" cofactor is $\frac{a^{2m}-b^{2m}}{a^m-b^m} = a^m + b^m$, which is itself even (odd $+$ odd), so the per-step count of one is wrong. Peel by repeated difference of squares: writing $n = 2^s t$ with $t$ odd,
$$a^n - b^n = (a^t - b^t)(a^t + b^t)\prod_{j=1}^{s-1}\bigl(a^{2^j t} + b^{2^j t}\bigr).$$
Here $v_2(a^t - b^t) = v_2(a-b)$ and $v_2(a^t + b^t) = v_2(a+b)$ ($t$ odd) because the corresponding quotients are odd, and each $a^{2^j t} + b^{2^j t}$ ($j \ge 1$) is a sum of two odd squares $\equiv 1 + 1 = 2 \pmod 4$, contributing exactly $1$; there are $s - 1$ of them. Summing: $v_2(a-b) + v_2(a+b) + (s-1) = v_2(a-b) + v_2(a+b) + v_2(n) - 1$.

## The anchor solved

Largest power of $2$ dividing $3^{2024} - 1$: take $a = 3$, $b = 1$ (both odd), $n = 2024$ (even). Since $a - b = 2$, we have $2 \,\|\, (a-b)$ (not $4 \mid a - b$), so the correction matters. With $v_2(2024) = 3$ (because $2024 = 2^3 \cdot 253$),
$$v_2(3^{2024} - 1) = v_2(3 - 1) + v_2(3 + 1) + v_2(2024) - 1 = 1 + 2 + 3 - 1 = 5.$$
Hence $2^5 \mid 3^{2024} - 1$ and $2^6 \nmid 3^{2024} - 1$: the largest power of $2$ dividing $3^{2024} - 1$ is $\boxed{2^5 = 32}$.
