OK, let me think this through from scratch. I have a finite set $A$ of $n$ nonzero integers, and I want a large subset with no solution to $x+y=z$. If I try to kill additive triples one at a time, I lose control immediately: a set can have quadratically many triples and the triples can overlap in any pattern. I need a rule that keeps a definite fraction of the elements and makes sum-freeness automatic.

Automatic sum-freeness suggests mapping the integers into a group and keeping the preimage of a region $B$ with $B+B$ disjoint from $B$. Then if $a+b=c$ and both $a,b$ land in $B$, the image of $c$ lands in $B+B$, so it cannot also land in $B$. The circle $\mathbb{T}=\mathbb{R}/\mathbb{Z}$ is the natural target, because $a\mapsto \theta a\bmod 1$ is additive for every $\theta$.

What is the largest simple region on the circle whose sumset misses itself? I try a symmetric arc $(\alpha,1-\alpha)$ around $1/2$. Two points in it have sum in $(2\alpha,2-2\alpha)$; modulo $1$ this wraps through $0$. To keep the wrapped sum away from the middle arc, I need the lower end $2\alpha$ to be at least the right end $1-\alpha$, so $\alpha\ge1/3$. The densest such symmetric arc is therefore $B=(1/3,2/3)$, with measure $1/3$. Checking directly, if $u,v\in(1/3,2/3)$ then $u+v\in(2/3,4/3)$, which reduces modulo $1$ into $(2/3,1)\cup(0,1/3)$ and misses $B$.

So I pick $\theta$ and keep
$$A_\theta=\{a\in A:\{\theta a\}\in(1/3,2/3)\}.$$
For every fixed $\theta$, this is sum-free. If $a+b=c$ with $a,b,c\in A_\theta$, then $\{\theta c\}=\{\theta a\}+\{\theta b\}$ in $\mathbb{T}$ would lie outside the middle third, contradicting $c\in A_\theta$.

Now the size is easy to average. For each nonzero integer $a$, the map $\theta\mapsto \theta a$ preserves Haar measure on $\mathbb{T}$, so
$$\Pr_\theta(\{\theta a\}\in(1/3,2/3))=\frac13.$$
Linearity of expectation gives
$$\mathbb{E}_\theta |A_\theta|=\sum_{a\in A}\Pr_\theta(\{\theta a\}\in(1/3,2/3))=\frac n3.$$
Some $\theta$ has $|A_\theta|\ge n/3$, and hence $s(A)\ge n/3$. The averaging is doing something stronger than it first looks like, because the selected size is always an integer. To turn that into a strict gain, I need to make sure the selector is not constant.

If I use the half-open arc $[1/3,2/3)$, which is still sum-free because the endpoint $2/3$ is excluded, then I can write
$$f(x)=1_{[1/3,2/3)}(x)-\frac13,\qquad f_A(x)=\sum_{a\in A}f(ax),$$
so $|A_x|=n/3+f_A(x)$ and $\int f_A=0$. At $x=0$ no element lies in $[1/3,2/3)$, so the selected set is empty; since the average is $n/3>0$, the integer-valued function $|A_x|$ is not constant. Its maximum is strictly larger than $n/3$, so
$$s(A)\ge \lfloor n/3\rfloor+1=\left\lceil\frac{n+1}{3}\right\rceil\ge\frac{n+1}{3}.$$
That is the Alon-Kleitman bump. It is real, but it is only one integer jump, so if I want another jump I need to understand how large $m_A=\max_x f_A(x)$ can be.

The function $f$ is simple enough to Fourier expand. Its cosine coefficient is
$$b_m=2\int_{1/3}^{2/3}\cos(2\pi m x)\,dx=\frac1{\pi m}\left(\sin\frac{4\pi m}{3}-\sin\frac{2\pi m}{3}\right).$$
If $m\equiv1\pmod3$, this is $-\sqrt3/(\pi m)$; if $m\equiv2\pmod3$, this is $+\sqrt3/(\pi m)$; and if $3\mid m$, it vanishes. Thus, with the nonprincipal character $\chi$ modulo $3$,
$$f(x)=-\frac{\sqrt3}{\pi}\sum_{m\ge1}\frac{\chi(m)}{m}\cos(2\pi m x),\qquad \chi(1)=1,\ \chi(2)=-1,\ \chi(3k)=0.$$
So
$$f_A(x)=-\frac{\sqrt3}{\pi}F_A(x),\qquad F_A(x)=\sum_{a\in A}\sum_{m\ge1}\frac{\chi(m)}{m}\cos(2\pi m a x).$$
This is the right object for the surplus. Since $F_A$ has mean zero,
$$\|F_A\|_1=2\int \max(-F_A(x),0)\,dx\le 2\max_x(-F_A(x)),$$
and therefore a large $L^1$ norm for $F_A$ immediately pushes $m_A$ above zero. The first harmonic inside $F_A$ is the plain Littlewood sum
$$c_A(x)=\sum_{a\in A}\cos(2\pi ax),$$
but I cannot simply throw away the rest of the series; the $1/m$ tail can cancel with it. I need to sift. If I Mobius-invert over primes up to $Q$, the multiplicativity of $\chi$ leaves
$$\sum_{k\mid \prod_{p\le Q}p}\frac{\mu(k)\chi(k)}k F_A(kx)=c_A(x)+R_Q(x),$$
where $R_Q$ contains only $Q$-rough harmonics with $m>1$. The triangle inequality costs
$$\prod_{p\le Q}\left(1+\frac1p\right)\ll\log Q,$$
and choosing $Q$ around $n^2$ makes $R_Q$ small in $L^1$ by Parseval. So the plain-cosine estimate that comes out is
$$s(A)\ge \frac n3+c\frac{\|c_A\|_{L^1(\mathbb{T})}}{\log n}.$$
That handles the sets with a large Littlewood norm. The sets that remain dangerous are structured, so I need a way to certify $m_A$ by hand.

A lower bound on a maximum can be certified by a nonnegative average. If $\varphi\ge0$ and $\int\varphi=1$, then
$$m_A=\max_x f_A(x)\ge \int_0^1 \varphi(x)f_A(x)\,dx.$$
If $\varphi(x)=c_0+\sum_{r\ge1}c_r\cos(2\pi r x)$, orthogonality gives
$$\int_0^1 \varphi f_A=-\frac{\sqrt3}{2\pi}\sum_{a\in A}\sum_{m\ge1}\frac{\chi(m)}{m}c_{ma}.$$
So the problem becomes concrete: build a nonnegative trigonometric polynomial of mean $1$, and make the finite set of frequencies in $\varphi$ interact with $A$ in a favorable way.

Products of $1-\cos(2\pi ux)$ are perfect for this because they are nonnegative. Suppose first that $A$ has coprime positive elements and $1\notin A$. Let $u=\min A$, and let $v$ be the smallest element of $A$ not divisible by $u$. I take
$$\varphi(x)=(1-\cos 2\pi ux)(1-\cos 2\pi vx).$$
Since $u\ne v$, its mean is $1$, and its nonzero cosine coefficients are
$$c_u=c_v=-1,\qquad c_{u+v}=\frac12,\qquad c_{v-u}=\frac12.$$
The coefficient at $u$ sees only $ma=u$, hence only $(m,a)=(1,u)$; any other solution would have $a<u$. The coefficient at $v$ sees $(1,v)$, and any solution with $m>1$ has $a<v$, so the choice of $v$ forces $u\mid a$ and then $u\mid v$, contrary to the definition of $v$. The frequency $v-u$ contributes nothing by the same minimality argument. At $u+v$, the solution with $m=1$ is just the possible element $u+v\in A$, while any solution with $m>1$ has $a\le(u+v)/2<v$ and again forces $u\mid v$. The value is therefore
$$\int\varphi f_A=\frac{\sqrt3}{2\pi}\left(2-\frac12 1_A(u+v)\right)\ge \frac{\sqrt3}{2\pi}\cdot\frac32>\frac13.$$
This is a clean certificate: whenever $1\notin A$, $m_A>1/3$.

The sets containing $1$ are exactly where this certificate stops being enough, and the mod-$3$ structure of $f$ becomes useful. I split
$$A=A_0\sqcup A_1,$$
where $A_0$ is the set of elements coprime to $3$ and $A_1$ is the set of multiples of $3$. If $a\in A_1$, then $f(ax)$ is unchanged by $x\mapsto x+1/3$ because $a/3$ is an integer. If $a\in A_0$, then the three points $ax$, $a(x+1/3)$, and $a(x+2/3)$ are separated by nonzero thirds and therefore hit the three third-intervals once each. With the half-open convention, exactly one of the three lies in $[1/3,2/3)$, so
$$f(ax)+f(a(x+1/3))+f(a(x+2/3))=0.$$
Summing over $A$ gives the descent identity
$$f_A(x)+f_A(x+1/3)+f_A(x+2/3)=3f_{A_1}(x).$$
If $x$ maximizes $f_{A_1}$, the left side is a sum of three values each at most $m_A$, so
$$3m_A\ge3m_{A_1},\qquad m_A\ge m_{A_1}.$$
The multiples of $3$ form a smaller copy of the same problem after division by $3$.

I also need a lower bound from the coprime-to-$3$ part. Let $r_1$ and $r_2$ be the numbers of elements of $A_0$ congruent to $1$ and $2$ modulo $3$. If $r_1\ge r_2$, then at $x=1/3$ the residue-$1$ elements contribute $2/3$ and the residue-$2$ elements contribute $-1/3$, so
$$f_{A_0}(1/3)=\frac23r_1-\frac13r_2=r_1-\frac13|A_0|\ge\frac16|A_0|.$$
If $r_2\ge r_1$, the same argument at $x=2/3$ gives $f_{A_0}(2/3)\ge |A_0|/6$. Multiples of $3$ sit at $0$ at both points, where $f(0)=-1/3$. Therefore
$$m_A\ge \max\{f_A(1/3),f_A(2/3)\}\ge \frac{|A_0|}{6}-\frac{|A_1|}{3}.$$

Now the induction has a shape. Suppose I want $m_A\ge S/3$ for all sufficiently large $n$. If
$$\frac{|A_0|}{6}-\frac{|A_1|}{3}\ge\frac S3,$$
the residue bound wins. Otherwise, using $|A_0|+|A_1|=n$, I get
$$|A_1|>\frac{n-2S}{3}.$$
So unless the residue classes already provide the surplus, the divisible-by-$3$ part is large enough to recurse on. With the descent $m_A\ge m_{A_1}$, proving the conjecture for a finite range $N_S<n\le 3N_S+2S$ propagates it to all larger $n$.

For the concrete bound $s(A)\ge(n+2)/3$, the previous Alon-Kleitman step already handles $n\equiv0,1\pmod3$ after integer rounding. The binding residue is $n\equiv2\pmod3$. In that residue class the possible positive values of $m_A$ begin at $1/3$ and then jump by integers, so proving $m_A>1/3$ is enough to force the next value and hence the required integer size. The finite descent leaves the cases $n=5$ and $n=8$.

For $n=5$, the descent leaves only the two-element obstruction inside the divisible-by-$3$ part, so I may reduce to $A_1=\{v,2v\}$. If $1\notin A$, the product certificate already gives $m_A>1/3$, so I may suppose $1\in A$. I try the nonnegative polynomial
$$\varphi(x)=1-\frac43\cos(2\pi x)+\frac23\cos(4\pi x)=\frac13(2\cos(2\pi x)-1)^2.$$
Its integral against $f_A$ is
$$\frac{\sqrt3}{2\pi}\left(\frac53-\frac23 1_A(2)\right).$$
If $2\notin A$, this is already larger than $1/3$, so I may suppose $2\in A$. Calling the remaining coprime-to-$3$ element $u$ and applying the same polynomial at frequency $u$, the coefficient calculation gives
$$\int f_A(x)\varphi(ux)\,dx=\frac{\sqrt3}{2\pi}\left(\frac53+\frac{\chi(u)}u-\frac{8\chi(u)}{3u}1_{u\equiv0\pmod2}\right)> \frac13,$$
using that $2u$ is not one of the available elements. The size-$5$ case is done.

For $n=8$, any larger divisible-by-$3$ subproblem would already be settled by the size-$5$ case through the descent, so I am again forced into $A_1=\{v,2v\}$ and $1,2\in A$. Let $u$ be the next element not among $1,2,v,2v$; otherwise the size-$5$ certificate applied to the visible five elements would already finish, so I may also suppose $2u\in A$. I look at $x=1/(6v)$. The divisible part contributes
$$f(vx)+f(2vx)=f(1/6)+f(1/3)=-\frac13+\frac23=\frac13.$$
The three-shift identity for $A_0$ implies that I already get $m_A>1/3$ unless the $A_0$ values at the three shifts are all exactly balanced at zero. Since $f(z)=-1/3$ for $0\le z<1/3$, the known elements $1,2,u,2u$ force one of the three shifted $A_0$ sums to become positive if either unknown element lies at most $2v$. The only hard case has the two unknown elements larger than $2v$. Then the product
$$\varphi(x)=(1-\cos(2\pi x))(1-\cos(2\pi vx))$$
has a manageable coefficient calculation. The main terms give
$$\int f_A\varphi\ge \frac{\sqrt3}{2\pi}\left(2-\frac18-\frac12 E_A\right),$$
where
$$E_A=\sum_{j=1}^2\sum_{n\ge1}\frac{\chi(n)}n\left(1_{nju=v+1}+1_{nju=v-1}\right).$$
The two divisibility alternatives $u\mid v+1$ and $u\mid v-1$ cannot both occur when $u>3$, so $E_A\le1/2$, and the right side is still $>1/3$. That finishes the finite cases. Together with the descent, this gives the theorem: for coprime positive $A$, either $A=\{1,2\}$ or $s(A)\ge(n+2)/3$.

The counting problem has a different flavor. The lower-bound families are too large to ignore: all subsets of the odds and all subsets of the strict upper half already give about $2^{N/2}$. The known upper bound $2^{N/2+o(N)}$ has the right exponent, but the $2^{o(N)}$ factor is still enormous compared with the constant. I need to cover all sum-free sets by few containers, and the containers have to be structured enough that their internal sum-free subsets can be counted sharply.

Exact sum-free containers would be too numerous, so I relax to almost sum-free containers: sets with $o(N^2)$ additive triples. I embed $[N]$ into $\mathbb{Z}/p\mathbb{Z}$ with $p\in[2N,4N]$ prime, so no equation $x+y=z$ from $[N]$ wraps around. Then I partition the group into $M$ arithmetic progressions $I_i$ of common difference $d$ and length $L=\lceil p/M\rceil$ or $L-1$. For a parameter $\epsilon_1$, I keep the dense blocks
$$T=\{i:|A\cap I_i|\ge \epsilon_1|I_i|\}$$
and define the granularization
$$A'=\bigcup_{i\in T} I_i.$$
The sparse blocks contain fewer than $\epsilon_1p$ points of $A$ in total, so $|A\setminus A'|\le\epsilon_1p$. That is the direction I need: $A'$ covers almost all of $A$, and a final container can add the missing few points back.

The difference $d$ cannot be arbitrary, because whole blocks can create new additive structure. I choose $d$ to respect the large Fourier coefficients of $A$. Let $|A|=\alpha p$, set
$$\delta=\frac1{16}\epsilon_1^2\epsilon_2\epsilon_3^{1/2}\alpha^{-1/2},$$
and let $R=\{r\ne0:|\widehat A(r)|\ge\delta p\}$. I call $d$ good when
$$\left\|\frac{dr}{p}\right\|\le \frac1{4L}\left(\frac{\delta p}{|\widehat A(r)|}\right)^{1/2}\qquad(r\in R).$$
The smoothing kernel
$$g(x)=\frac1{2L-1}\sum_{|j|<L}e(jdx/p)$$
satisfies, using $1-\cos 2\pi t\le2\pi^2\|t\|^2$,
$$1-g(x)=\frac2{2L-1}\sum_{j=1}^{L-1}\left(1-\cos\frac{2\pi jdx}{p}\right)
\le \frac{2\pi^2L^2}{3}\left\|\frac{dx}{p}\right\|^2.$$
Hence
$$|\widehat A(x)|\,|1-g(x)^2|\le 14L^2\left\|\frac{dx}{p}\right\|^2|\widehat A(x)|,$$
and the good-length inequality makes this at most $\delta p$ for the large coefficients; outside $R$, the bound $|\widehat A(x)|<\delta p$ and $|g(x)|\le1$ give the same conclusion. A pigeonhole/Dirichlet argument plus Parseval and AM-GM gives such a $d$ once
$$p>(4L)^{256\alpha^2\epsilon_1^{-4}\epsilon_2^{-2}\epsilon_3^{-1}}.$$

Now I smooth $A$ by the progression $P=\{-(L-1)d,\dots,(L-1)d\}$:
$$a_1(n)=\frac1{|P|}(A*P)(n),\qquad \widehat{a_1}(x)=\widehat A(x)g(x).$$
Two applications of Parseval give
$$\sum_n |(A*A)(n)-(a_1*a_1)(n)|^2\le \alpha\delta^2p^3.$$
If $n\in A'$, then $a_1(n)\ge \epsilon_1/4$, so
$$a_1*a_1(n)\ge \frac{\epsilon_1^2}{16}\,A'*A'(n).$$
Therefore all but at most $\epsilon_3p$ points with $A'*A'(n)\ge\epsilon_2p$ already lie in $A+A$.

If $A$ is sum-free and $A'$ had too many additive triples, a block-level double count would produce many triples of dense blocks with $i+j=k$ or $i+j=k+1$. In the middle of many target blocks, this forces $A'*A'(z)\ge\epsilon^2p/144$, while density of the target block supplies many actual points of $A$. Taking
$$\epsilon_1=\epsilon,\qquad \epsilon_2=\epsilon^2/144,\qquad \epsilon_3=\epsilon^2/80,$$
the previous Fourier transfer would then put a point of $A$ inside $A+A$, contradicting sum-freeness. Thus the granularization of a sum-free set has at most $\epsilon p^2$ additive triples.

I choose
$$\epsilon=(\log N)^{-1/11},\qquad M=\left\lfloor N\exp(-(\log N)^{1/12})\right\rfloor.$$
For large $N$, good lengths exist. I form a family by taking all unions of the progression blocks, discarding those with more than $\epsilon p^2$ triples, intersecting with $[N]$, and then allowing at most $\epsilon p$ extra elements. There are at most $p2^M$ block unions and only $2^{o(N)}$ ways to add the extra elements, so the family size is $2^{o(N)}$. Every sum-free subset of $[N]$ is contained in one of these almost-sum-free containers.

The remaining task is structural. Suppose $A\subseteq[N]$ has at most $\epsilon N^2$ triples and $|A|=(1/2-\eta)N$ with $\eta\le1/50$. I define the popular-difference set
$$D(A,K)=\{d:\#\{(a,a')\in A^2:a-a'=d\}\ge K\}.$$
If too many popular differences also lay in $A$, each such $d\in A$ would create many triples $a'+d=a$. This gives
$$\frac12|D(A,\epsilon^{1/2}N)|+|A|\le N(1+2\epsilon^{1/2}).$$
So a large almost-sum-free set has a surprisingly small popular-difference set.

A graph cleanup says that, for all but $8\epsilon^{1/4}N$ elements $a\in A$, at least $|A|-16\epsilon^{1/4}N$ of the differences $a-a'$ are popular at threshold $32\epsilon^{1/2}N$. I choose near-left and near-right points $m$ and $M$ so that most elements have both $a-m$ and $a-M$ popular, and most of $A$ lies in $(m,M)$. Let $t=M-m$; the construction gives $t>N/4$. Projecting modulo $t$ by $\pi:\mathbb{Z}\to\mathbb{Z}/t\mathbb{Z}$ is at most four-to-one on $[N]$, so popular differences mostly survive projection. Since $a-m\equiv a-M\pmod t$, many projected popular differences have two popular lifts, improving the count to
$$|D(\pi(A),8\epsilon^{1/2}N)|\le 2N(1+30\epsilon^{1/8})-3|A|.$$

The Lev-Luczak-Schoen graph lemma then gives a large subset $X\subseteq\pi(A)$ with
$$|X|\ge |A|-54\epsilon^{1/8}N,\qquad X-X\subseteq D(\pi(A),8\epsilon^{1/2}N).$$
Now the Kneser alternatives take over. If $|X|>t/2$ then $X-X=\mathbb{Z}/t\mathbb{Z}$, and the popular-difference bound forces
$$t\le \left(\frac12+3\eta+60\epsilon^{1/8}\right)N,$$
so almost all of $A$ lies in an interval of that length. If $|X-X|\ge 2|X|-t/3$, the same bounds imply $|A|\le(7/15+O(\epsilon^{1/8}))N$, contradicting the assumed density. If $|X-X|<2|X|-t/3$, Kneser's theorem forces $X-X$ to be a union of cosets of a subgroup of index $2$ in $\mathbb{Z}/t\mathbb{Z}$. Thus $t$ is even and almost all of $A$ has one parity. The even-parity alternative would put at least $12N/25$ elements inside the evens, creating at least $N^2/100$ additive triples, so the parity must be odd. A large almost-sum-free set is therefore essentially contained in a short interval or is essentially all odd.

This already recovers the exponential estimate $2^{N/2+o(N)}$, because an almost-sum-free container cannot have size much above $N/2$. To remove the $o(N)$ in the exponent, I count the large-container cases more carefully. Containers of size at most $(1/2-1/120)N$ contribute
$$2^{o(N)}2^{(1/2-1/120)N}=o(2^{N/2}).$$
If the container is large and essentially odd, then any sum-free subset with even element $t$ loses choices: for $t<N/2$ I can find $\lfloor N/8\rfloor$ disjoint pairs $(x,x+t)$ of odd numbers, and for $t\ge N/2$ I use pairs $(x,t-x)$. The set cannot contain both elements of any pair, so the number of such choices is at most
$$2^{N/4+o(N)}3^{N/8}=o(2^{N/2}).$$
Thus almost all sets in the odd case are entirely odd.

If the container is essentially interval-like and a sum-free set contains some
$$t\le \left\lfloor\frac{N+1}{3}\right\rfloor,$$
then I can choose about $N/12$ disjoint pairs $(x,x+t)$ with $x\ge N/2$, and again at most one from each pair may appear. The number of such sets is bounded by
$$3^{N/12}2^{(1/3+1/30+o(1))N}=o(2^{N/2}).$$
So, up to $o(2^{N/2})$ exceptions, every sum-free subset is either entirely odd or is contained in
$$\left\{\left\lceil\frac{N+1}{3}\right\rceil,\dots,N\right\}.$$
The count for that interval is $\sim c_0(N)2^{N/2}$ with two parity-dependent constants; adding the entirely odd family and subtracting the overlap only changes the parity-dependent constant. Therefore
$$|\mathrm{SF}(N)|\sim c(N)2^{N/2}.$$

The route has landed on two mechanisms working together: a one-third sum-free arc gives the averaging bound, integrality forces the first integer improvement, Fourier exposes the surplus and supplies nonnegative certificates for structured cases, mod-$3$ descent reduces the remaining obstruction to finite checks, and granularized almost-sum-free containers plus the popular-difference/Kneser dichotomy reduce the enumeration to the odd and interval families.
