I will solve Putnam 2020 B6 and call the method **Pell self-similarity for Beatty sign sums**. Let $\alpha=\sqrt2-1$ and define $f(n)=\sum_{k=1}^n(-1)^{\lfloor k\alpha\rfloor}$ with $f(0)=0$. The exponent $\lfloor k\alpha\rfloor$ is a staircase in $k$. Because $0<\alpha<1$, moving from $k$ to $k+1$ either leaves the floor unchanged or increases it by exactly one. So the signs split into runs: while $\lfloor k\alpha\rfloor=m$ the sign is constant, and it flips whenever $m$ increases. The block on which $\lfloor k\alpha\rfloor=m$ is determined by $m\beta\le k<(m+1)\beta$, where $\beta=1/\alpha=\sqrt2+1$. The number of positive integers in block $m$ is $L_m=\lfloor(m+1)\beta\rfloor-\lfloor m\beta\rfloor$. Since $\beta$ lies strictly between $2$ and $3$ and is irrational, every $L_m$ is exactly $2$ or $3$. Thus the signs alternate between positive and negative runs, each of length two or three, starting with a positive run.

Because $f$ increases on positive runs and decreases on negative runs, its minima can only occur at run boundaries. After blocks $0,1,\dots,M-1$ the boundary is at $N_M=L_0+\cdots+L_{M-1}=\lfloor M\beta\rfloor$, and the value there is $f(N_M)=\sum_{m=0}^{M-1}(-1)^mL_m$. So the whole problem reduces to showing that this Beatty-driven alternating walk, whose step lengths are always $2$ or $3$, never drops below zero. Knowing only that each run has length $2$ or $3$ is not enough; the order of the lengths matters, and that order is governed by the continued fraction of $\alpha$.

The continued fraction $\alpha=[0;2,2,2,\dots]$ leads to the Pell denominators $q_0=0$, $q_1=1$, and $q_j=2q_{j-1}+q_{j-2}$ for $j\ge2$. They satisfy $q_j\equiv j\pmod2$ and the exact approximation identity $q_j\alpha-q_{j-1}=(-1)^{j+1}\alpha^j$. Consecutive convergents are extremely close to each other: for $1\le r<q_{j+1}$, the distance $\|r\alpha\|$ from $r\alpha$ to the nearest integer is at least $\alpha^j$. From these two facts I derive self-similar recursions for $f$ on each interval $q_j\le n<q_j+q_{j+1}$.

When $j$ is odd, $q_{j-1}$ is even and $q_{j-1}/q_j<\alpha$. For every $k<q_j+q_{j+1}$ the irrational floor $\lfloor k\alpha\rfloor$ equals the rational floor $\lfloor kq_{j-1}/q_j\rfloor$. Shifting a window of length $q_j$ does not change the sign pattern because adding $q_j$ adds the even integer $q_{j-1}$ to the floor. Pairing $k$ and $q_j-k$ inside one period leaves only the central term, whose sign is $(-1)^{q_{j-1}}=+1$. Hence $f(n)=f(n-q_j)+1$ for $q_j\le n<q_j+q_{j+1}$.

When $j$ is even, $q_{j-1}$ is odd and $q_j\alpha=q_{j-1}-\eta$ with $\eta=\alpha^j>0$. The closest-return property keeps the fractional part of $r\alpha$ at least $\eta$ away from zero for $r<q_{j+1}$, so adding $q_j$ to $r$ subtracts $\eta$ without crossing an integer. Therefore $\lfloor(q_j+r)\alpha\rfloor=q_{j-1}+\lfloor r\alpha\rfloor$, and because $q_{j-1}$ is odd the sign flips. This gives $f(n)=f(q_j)-f(n-q_j)$ on the same interval.

These recursions make the even Pell peaks easy to compute. Applying the odd recursion twice with $j=2t-1$ to $n=q_{2t}$ and to $q_{2t}-q_{2t-1}$, and using the Pell recurrence $q_{2t}-2q_{2t-1}=q_{2t-2}$, yields $f(q_{2t})=f(q_{2t-2})+2$. Starting from $f(q_0)=0$, the peaks are $f(q_{2t})=2t$.

I now prove $0\le f(m)\le 2t$ for all $0\le m\le q_{2t}$ by induction on $t$. The base $t=1$ is $f(0)=0$, $f(1)=1$, and $f(2)=2$. Assume the bound holds up to $q_{2t}$. First I extend it to every $m<q_{2t+1}$. If $m\le q_{2t}$ there is nothing to show. If $q_{2t}<m<q_{2t+1}$, write $m=q_{2t}+r$. The even recursion gives $f(m)=2t-f(r)$. If $r\le q_{2t}$ the induction hypothesis bounds $f(r)$, so $0\le f(m)\le2t$. If $r>q_{2t}$, I apply the even recursion again to $r$, using $r<q_{2t}+q_{2t-1}<q_{2t}+q_{2t+1}$; this gives $f(r)=2t-f(r-q_{2t})$, and substituting gives $f(m)=f(r-q_{2t})$, which is again between $0$ and $2t$.

Next consider $q_{2t+1}\le m\le q_{2t+2}$ and write $m=q_{2t+1}+r$. The odd recursion gives $f(m)=f(r)+1$. If $r<q_{2t+1}$, the previous step gives $0\le f(r)\le2t$, so $1\le f(m)\le2t+1$. If $r\ge q_{2t+1}$, I apply the odd recursion once more to obtain $f(r)=f(r-q_{2t+1})+1$ with $r-q_{2t+1}\le q_{2t}$; then $2\le f(m)\le2t+2$. This completes the induction step up to $q_{2t+2}$.

Because the even Pell numbers grow without bound, every positive integer $n$ lies below some $q_{2t}$, and the induction guarantees $f(n)\ge0$. That is exactly the desired inequality $\sum_{k=1}^n(-1)^{\lfloor k(\sqrt2-1)\rfloor}\ge0$.

The method is conceptually clean: the sign pattern is a Beatty sequence of runs whose lengths are always $2$ or $3$, and the Pell denominators capture the self-similarity of that pattern. The two recursions, one lifting by one on odd scales and one reflecting on even scales, are enough to bound the partial sums inductively. The following Python script checks the original inequality, the Pell recursions, the even Pell peaks, and the induction bound numerically.

```python
import math


def verify_putnam_2020_b6(N=20000):
    alpha = math.sqrt(2) - 1.0

    # Partial sums f(n) for n = 0, 1, ..., N.
    f = [0] * (N + 1)
    for k in range(1, N + 1):
        sign = -1 if int(math.floor(k * alpha)) % 2 else 1
        f[k] = f[k - 1] + sign

    # Direct nonnegativity check.
    assert min(f[1:]) >= 0, "partial sum became negative"

    # Pell denominators q_j up to exceeding N.
    q = [0, 1]
    while q[-1] <= N:
        q.append(2 * q[-1] + q[-2])

    # Verify even Pell peaks f(q_{2t}) == 2t.
    for t in range(1, len(q) // 2):
        if q[2 * t] > N:
            break
        assert f[q[2 * t]] == 2 * t, f"peak mismatch at t={t}"

    # Verify the self-similar recursions on their intervals.
    for j in range(1, len(q) - 1):
        qj = q[j]
        qj1 = q[j + 1]
        if qj > N:
            break
        upper = min(qj + qj1, N + 1)  # n must satisfy q_j <= n < q_j + q_{j+1}
        for n in range(qj, upper):
            r = n - qj
            if j % 2 == 1:
                expected = f[r] + 1
            else:
                expected = f[qj] - f[r]
            assert f[n] == expected, f"recursion failed at j={j}, n={n}"

    # Verify the induction bound 0 <= f(m) <= 2t for all m <= q_{2t}.
    for t in range(1, (len(q) - 1) // 2):
        q2t = q[2 * t]
        if q2t > N:
            break
        assert all(0 <= f[m] <= 2 * t for m in range(q2t + 1)), \
            f"induction bound failed at t={t}"

    print(f"All checks passed up to n={N}.")


if __name__ == "__main__":
    verify_putnam_2020_b6()
```
