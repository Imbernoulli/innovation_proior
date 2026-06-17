# Theorem

For every positive integer $n$,
$$
\sum_{k=1}^{n}(-1)^{\lfloor k(\sqrt2-1)\rfloor}\ge0.
$$

# Proof

Let
$$
\alpha=\sqrt2-1,\qquad \beta={1\over\alpha}=\sqrt2+1,\qquad
f(n)=\sum_{k=1}^{n}(-1)^{\lfloor k\alpha\rfloor},\quad f(0)=0.
$$

For each $m\ge0$, the integers $k$ for which $\lfloor k\alpha\rfloor=m$ are those satisfying
$$
m\beta\le k<(m+1)\beta.
$$
Hence the length of this sign run is
$$
L_m=\lfloor(m+1)\beta\rfloor-\lfloor m\beta\rfloor.
$$
Since $\beta\in(2,3)$ is irrational, $L_m\in\{2,3\}$ for all $m$. The signs are constant on each such block and alternate from block to block. Therefore $f$ increases on positive runs and decreases on negative runs; its minima occur at run boundaries
$$
N_M=L_0+\cdots+L_{M-1}=\lfloor M\beta\rfloor,
$$
where
$$
f(N_M)=\sum_{m=0}^{M-1}(-1)^mL_m.
$$
The run-length picture reduces the problem to controlling this Beatty-ordered alternating walk. The control comes from the Pell self-similarity below.

Define
$$
q_0=0,\qquad q_1=1,\qquad q_j=2q_{j-1}+q_{j-2}\quad(j\ge2).
$$
Then $q_j\equiv j\pmod2$, and for $j\ge1$, $q_{j-1}/q_j$ is the $j$th convergent to $\alpha$. Also, for $j\ge1$,
$$
q_j\alpha-q_{j-1}=(-1)^{j+1}\alpha^j. \tag{1}
$$
Consecutive convergents give two standard consequences used below. No rational, after reduction, with denominator less than $q_j+q_{j+1}$ lies between $q_{j-1}/q_j$ and $q_j/q_{j+1}$. Also, for $1\le r<q_{j+1}$,
$$
\|r\alpha\|\ge |q_j\alpha-q_{j-1}|=\alpha^j. \tag{2}
$$

We prove the self-similarity formulas for $j\ge1$. If $q_j\le n<q_j+q_{j+1}$, then
$$
f(n)=
\begin{cases}
f(n-q_j)+1, & j\text{ odd},\\
f(q_j)-f(n-q_j), & j\text{ even}.
\end{cases} \tag{3}
$$

First suppose $j$ is odd. Put $q=q_j$ and $p=q_{j-1}$; then $p$ is even and $p/q<\alpha$. For $1\le k<q+q_{j+1}$,
$$
\lfloor k\alpha\rfloor=\left\lfloor {kp\over q}\right\rfloor,
$$
because otherwise some integer $a$ would satisfy $kp/q<a\le k\alpha$, giving a rational $a/k$ between $p/q$ and $\alpha$; after reducing it, its denominator is still less than $q+q_{j+1}$. If $n=q+r$ with $0\le r<q_{j+1}$, then
$$
f(n)-f(r)=\sum_{k=r+1}^{r+q}(-1)^{\lfloor kp/q\rfloor}.
$$
Since $p$ is even, the signs $(-1)^{\lfloor kp/q\rfloor}$ are periodic with period $q$. Over one period, the terms $k$ and $q-k$ cancel for $1\le k\le q-1$, because
$$
\left\lfloor {kp\over q}\right\rfloor+\left\lfloor {(q-k)p\over q}\right\rfloor=p-1
$$
is odd. The remaining term $k=q$ contributes $(-1)^p=1$. Hence $f(n)=f(n-q_j)+1$.

Now suppose $j$ is even. Put again $q=q_j$, $p=q_{j-1}$; now $p$ is odd and, by (1), $q\alpha=p-\eta$ with $\eta=\alpha^j>0$. For $1\le r<q_{j+1}$, write $r\alpha=a+\theta$ with $a=\lfloor r\alpha\rfloor$ and $0<\theta<1$. By (2), $\theta<\eta$ is impossible, so $\eta\le\theta<1$ and
$$
\lfloor(q+r)\alpha\rfloor
=\lfloor p+r\alpha-\eta\rfloor
=p+\lfloor r\alpha\rfloor.
$$
Because $p$ is odd,
$$
(-1)^{\lfloor(q+r)\alpha\rfloor}=-(-1)^{\lfloor r\alpha\rfloor}.
$$
Thus, for $n=q+r$ with $0\le r<q_{j+1}$, with the case $r=0$ contributing an empty shifted sum,
$$
f(n)=f(q)+\sum_{s=1}^{r}(-1)^{\lfloor(q+s)\alpha\rfloor}
     =f(q)-f(r)
     =f(q_j)-f(n-q_j).
$$
This proves (3).

Next compute the even Pell peaks. Applying the odd case of (3) twice, with $j=2t-1$, is valid because both $q_{2t}$ and $q_{2t}-q_{2t-1}$ lie in $[q_{2t-1},q_{2t-1}+q_{2t})$. It gives
$$
f(q_{2t})
=f(q_{2t}-q_{2t-1})+1
=f(q_{2t}-2q_{2t-1})+2
=f(q_{2t-2})+2.
$$
Since $f(q_0)=0$,
$$
f(q_{2t})=2t. \tag{4}
$$

We now prove by induction on $t\ge1$ that
$$
0\le f(m)\le2t\qquad(0\le m\le q_{2t}). \tag{5}
$$
For $t=1$, this is $f(0)=0$, $f(1)=1$, $f(2)=2$.

Assume (5) holds for $t$. First extend it to every $m<q_{2t+1}$. If $m\le q_{2t}$ there is nothing to show. If $q_{2t}<m<q_{2t+1}$, write $m=q_{2t}+r$; then $0<r<q_{2t}+q_{2t-1}$. The even case of (3) gives
$$
f(m)=f(q_{2t})-f(r)=2t-f(r).
$$
If $r\le q_{2t}$, the induction hypothesis gives $0\le f(m)\le2t$. If $r>q_{2t}$, then $r<q_{2t}+q_{2t-1}<q_{2t}+q_{2t+1}$, so the same even case applies once more to $r$:
$$
f(r)=2t-f(r-q_{2t}),
$$
where $0<r-q_{2t}<q_{2t}$. Then $f(m)=f(r-q_{2t})$, again between $0$ and $2t$.

Now let $q_{2t+1}\le m\le q_{2t+2}$ and write $m=q_{2t+1}+r$, so $0\le r\le q_{2t+1}+q_{2t}$. The odd case of (3) gives
$$
f(m)=f(r)+1.
$$
If $r<q_{2t+1}$, the preceding paragraph gives $0\le f(r)\le2t$, so $1\le f(m)\le2t+1$. If $r\ge q_{2t+1}$, then $r<q_{2t+1}+q_{2t+2}$, so the odd case applies once more:
$$
f(r)=f(r-q_{2t+1})+1,
$$
with $0\le r-q_{2t+1}\le q_{2t}$. By the induction hypothesis,
$$
2\le f(m)\le2t+2.
$$
Thus (5) holds with $t$ replaced by $t+1$.

Since $q_{2t}\to\infty$, every positive integer $n$ is covered by (5) for some $t$. Therefore $f(n)\ge0$ for all $n$, which is the desired inequality.
