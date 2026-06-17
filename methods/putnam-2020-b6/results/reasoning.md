I need to keep the partial sums of the signs
$$
\epsilon_k=(-1)^{\lfloor k(\sqrt 2-1)\rfloor}
$$
from ever going below zero. Let me name $\alpha=\sqrt 2-1$ and $f(n)=\sum_{k=1}^n\epsilon_k$, with $f(0)=0$. The exponent is the staircase $\lfloor k\alpha\rfloor$, and since $0<\alpha<1$, that staircase either stays put or rises by one when $k$ increases by one. So the signs are constant while the floor has one value, then they flip when the floor rises to the next value. This is really a question about the lengths of the consecutive sign runs.

The block on which $\lfloor k\alpha\rfloor=m$ is determined by
$$
m\le k\alpha<m+1.
$$
Since $1/\alpha=\sqrt 2+1$, writing $\beta=\sqrt 2+1$ turns this into
$$
m\beta\le k<(m+1)\beta.
$$
So the number of positive integers in block $m$ is
$$
L_m=\lfloor (m+1)\beta\rfloor-\lfloor m\beta\rfloor.
$$
The important point is not the exact pattern yet, only the range: $\beta\in(2,3)$, so each interval has length between $2$ and $3$, and because $\beta$ is irrational the count is always exactly $2$ or $3$. Thus the signs come in alternating runs, starting with a positive run, and every run has length $2$ or $3$.

That already localizes where danger can occur. Across a positive run, $f$ only rises. Across a negative run, $f$ only falls. Therefore the minimum inside a run is at an endpoint, and the only endpoints I have to watch are the run boundaries. After blocks $0,1,\ldots,M-1$, the boundary is at
$$
N_M=L_0+\cdots+L_{M-1}=\lfloor M\beta\rfloor,
$$
and the value there is
$$
P_M=\sum_{m=0}^{M-1}(-1)^mL_m=f(\lfloor M\beta\rfloor).
$$
So the Beatty run-length picture reduces the shape of the partial sums to an alternating sum of $2$'s and $3$'s. But just knowing "each run has length $2$ or $3$" is not enough: a short positive run followed by a long negative run can lose one unit. I need the specific order forced by $\sqrt 2$, not just the local run lengths.

The natural scale for $\sqrt 2-1=[0;2,2,2,\ldots]$ is the Pell scale. Let
$$
q_0=0,\qquad q_1=1,\qquad q_j=2q_{j-1}+q_{j-2}\quad(j\ge2),
$$
so
$$
0,1,2,5,12,29,70,\ldots
$$
are the denominators of the convergents $q_{j-1}/q_j$ to $\alpha$ for $j\ge1$. Two elementary facts about them are exactly what I need. First, $q_j\equiv j\pmod 2$, because the recurrence gives $q_j\equiv q_{j-2}\pmod 2$. Second, for $j\ge1$,
$$
q_j\alpha-q_{j-1}=(-1)^{j+1}\alpha^j.
$$
This is true for $j=1,2$, and the recurrence preserves it because $\alpha^2=1-2\alpha$. So when $j$ is odd, $q_j\alpha$ is just above the even integer $q_{j-1}$; when $j$ is even, $q_j\alpha$ is just below the odd integer $q_{j-1}$.

I also need the usual sharpness of consecutive convergents. Since $q_{j-1}/q_j$ and $q_j/q_{j+1}$ are neighboring convergents, their cross determinant has absolute value $1$, so no rational with denominator less than $q_j+q_{j+1}$ lies strictly between them. Equivalently, before the next denominator appears, the convergent gives the closest return: for $1\le r<q_{j+1}$,
$$
\|r\alpha\|\ge |q_j\alpha-q_{j-1}|=\alpha^j.
$$
Now I can turn the continued fraction information into exact recursions for $f$.

Take $j$ odd first. Then $p=q_{j-1}$ is even, $q=q_j$, and $p/q<\alpha$. For every $1\le k<q+q_{j+1}$ I claim
$$
\lfloor k\alpha\rfloor=\left\lfloor {kp\over q}\right\rfloor.
$$
If this failed, since $p/q<\alpha$, some integer $a$ would satisfy
$$
{kp\over q}<a\le k\alpha,
$$
so the rational $a/k$ would lie strictly between $p/q$ and $\alpha$, with denominator $k<q+q_{j+1}$. That contradicts the neighboring-convergent fact. Good: on this whole range the irrational signs are exactly the rational signs.

Now let $q_j\le n<q_j+q_{j+1}$ and write $n=q+r$, where $0\le r<q_{j+1}$. The difference $f(n)-f(r)$ is the sum of a window of length $q$:
$$
\sum_{k=r+1}^{r+q}(-1)^{\lfloor k\alpha\rfloor}.
$$
All indices in this window are below $q+q_{j+1}$, so I may replace $\lfloor k\alpha\rfloor$ by $\lfloor kp/q\rfloor$. Since $p$ is even,
$$
\left\lfloor{(k+q)p\over q}\right\rfloor=\left\lfloor{kp\over q}\right\rfloor+p
$$
has the same parity as $\lfloor kp/q\rfloor$. Thus any $q$ consecutive rational signs have the same sum as the period $k=1,\ldots,q$.

Inside that period, pair $k$ with $q-k$ for $1\le k\le q-1$. Because $\gcd(p,q)=1$, neither $kp/q$ nor $(q-k)p/q$ is an integer, and
$$
\left\lfloor{kp\over q}\right\rfloor+\left\lfloor{(q-k)p\over q}\right\rfloor=p-1.
$$
Here $p$ is even, so $p-1$ is odd, and the two paired signs cancel. The remaining term is $k=q$, whose sign is $(-1)^p=+1$. Therefore the whole window has sum $1$, and I get the odd-index recursion
$$
f(n)=f(n-q_j)+1\qquad(q_j\le n<q_j+q_{j+1},\ j\ {\rm odd}).
$$

Now take $j$ even. Then $p=q_{j-1}$ is odd, $q=q_j$, and
$$
q\alpha=p-\eta,\qquad \eta=\alpha^j>0.
$$
I want to see what shifting by $q$ does to each sign. For $1\le r<q_{j+1}$, write $r\alpha=a+\theta$ with $a=\lfloor r\alpha\rfloor$ and $0<\theta<1$. The closest-return property gives $\|r\alpha\|\ge\eta$, so in particular $\theta<\eta$ is impossible; equality would be harmless. Thus $\eta\le\theta<1$, and
$$
(q+r)\alpha=p+r\alpha-\eta=p+a+(\theta-\eta),
$$
with $0\le\theta-\eta<1$. Thus
$$
\lfloor(q+r)\alpha\rfloor=p+\lfloor r\alpha\rfloor.
$$
Since $p$ is odd, the sign flips:
$$
\epsilon_{q+r}=-\epsilon_r\qquad(1\le r<q_{j+1}).
$$
So for $q_j\le n<q_j+q_{j+1}$, writing $n=q+r$ gives
$$
f(n)=f(q)+\sum_{s=1}^{r}\epsilon_{q+s}
     =f(q)-\sum_{s=1}^{r}\epsilon_s
     =f(q_j)-f(n-q_j).
$$
That is the even-index recursion
$$
f(n)=f(q_j)-f(n-q_j)\qquad(q_j\le n<q_j+q_{j+1},\ j\ {\rm even}).
$$

Now I need to turn these two recursions into nonnegativity without losing a sign. The even-index peaks are easy once I use the odd recursion twice. For $t\ge1$, apply the odd recursion with $j=2t-1$ first to $n=q_{2t}$ and then to $n=q_{2t}-q_{2t-1}$. Both numbers lie in the interval
$$
q_{2t-1}\le n<q_{2t-1}+q_{2t}.
$$
Therefore
$$
f(q_{2t})=f(q_{2t}-q_{2t-1})+1
        =f(q_{2t}-2q_{2t-1})+2.
$$
The Pell recurrence gives $q_{2t}-2q_{2t-1}=q_{2t-2}$, so
$$
f(q_{2t})=f(q_{2t-2})+2.
$$
Starting from $f(q_0)=f(0)=0$, this gives
$$
f(q_{2t})=2t.
$$

The peaks suggest the right induction statement: for every $t\ge1$,
$$
0\le f(m)\le 2t\qquad(0\le m\le q_{2t}).
$$
The base $t=1$ is direct: $f(0)=0$, $f(1)=1$, and $f(2)=2$. Suppose the statement holds up to $q_{2t}$. I want to prove it up to $q_{2t+2}$.

First handle $m<q_{2t+1}$. If $m\le q_{2t}$, the induction hypothesis already applies. If $q_{2t}<m<q_{2t+1}$, write $m=q_{2t}+r$, where
$$
0<r<q_{2t+1}-q_{2t}=q_{2t}+q_{2t-1}.
$$
The even recursion at $j=2t$ gives
$$
f(m)=f(q_{2t})-f(r)=2t-f(r).
$$
If $r\le q_{2t}$, the induction hypothesis says $0\le f(r)\le2t$, so $0\le f(m)\le2t$. If $r>q_{2t}$, the same even recursion also applies to $r$, because
$$
q_{2t}<r<q_{2t}+q_{2t-1}<q_{2t}+q_{2t+1}.
$$
Then
$$
f(r)=2t-f(r-q_{2t}),
$$
and $0<r-q_{2t}<q_{2t-1}<q_{2t}$. Substituting back gives
$$
f(m)=f(r-q_{2t}),
$$
which is again between $0$ and $2t$ by the induction hypothesis. So the same bound, $0\le f(m)\le2t$, has now been extended all the way to $m<q_{2t+1}$.

Now handle the rest, $q_{2t+1}\le m\le q_{2t+2}$. Write $m=q_{2t+1}+r$, so
$$
0\le r\le q_{2t+2}-q_{2t+1}=q_{2t+1}+q_{2t}.
$$
The odd recursion at $j=2t+1$ gives
$$
f(m)=f(r)+1.
$$
If $r<q_{2t+1}$, I just proved $0\le f(r)\le2t$, so
$$
1\le f(m)\le2t+1.
$$
If $r\ge q_{2t+1}$, the odd recursion applies once more to $r$:
$$
f(r)=f(r-q_{2t+1})+1,
$$
and now $0\le r-q_{2t+1}\le q_{2t}$. The induction hypothesis gives
$$
0\le f(r-q_{2t+1})\le2t,
$$
so
$$
2\le f(m)\le2t+2.
$$
That proves
$$
0\le f(m)\le2t+2\qquad(0\le m\le q_{2t+2}),
$$
which is the induction step.

Since the even Pell numbers $q_{2t}$ grow without bound, every positive integer $n$ lies below some $q_{2t}$. The induction gives $f(n)\ge0$ for that $t$, which is exactly
$$
\sum_{k=1}^{n}(-1)^{\lfloor k(\sqrt2-1)\rfloor}\ge0.
$$
I can now see why the two pieces fit. The run-length analysis tells me that any possible dip is controlled by the ends of the negative runs, and the Pell recursions tell me that those ends are always reflected from a nonnegative smaller value or lifted by one. The signs in the two recursions are forced by parity: when $j$ is odd, $q_{j-1}$ is even and the length-$q_j$ rational period sums to $1$; when $j$ is even, $q_{j-1}$ is odd and shifting by $q_j$ flips every sign before the next Pell scale.
