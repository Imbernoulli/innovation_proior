# Topological Proof of the Infinitude of Primes

## Theorem

There are infinitely many prime numbers.

## Key idea

Put a topology on $\mathbb{Z}$ in which the arithmetic progressions are exactly the basic open sets. In this topology every nonempty open set is infinite, and each arithmetic progression is simultaneously open and closed. Since the non-units of $\mathbb{Z}$ are precisely the union over all primes $p$ of the multiples of $p$, assuming only finitely many primes would make that union a finite union of closed sets — hence closed — which would force the two-point set $\{-1,1\}$ to be open, contradicting that nonempty open sets are infinite.

## Construction

For $a\in\mathbb{Z}\setminus\{0\}$ and $b\in\mathbb{Z}$, define the two-sided arithmetic progression
$$S(a,b)=\{\,an+b : n\in\mathbb{Z}\,\}=a\mathbb{Z}+b.$$
Note $S(a,b)=S(-a,b)$, so one may always take $a>0$. Take the family $\mathcal{B}=\{S(a,b):a\neq 0\}$ as a basis and call a set **open** iff it is a union of members of $\mathcal{B}$ (the empty set being the empty union). This is the *evenly spaced integer topology*.

## Proof

**1. $\mathcal{B}$ is a basis (so this is a topology).**
$\mathcal{B}$ covers $\mathbb{Z}$, since $x\in S(1,x)=\mathbb{Z}$ for every $x$. For the intersection condition, suppose $x\in S(a_1,b_1)\cap S(a_2,b_2)$. Then $x\equiv b_i\pmod{a_i}$, so $S(a_i,b_i)=S(a_i,x)$. Let $a=\operatorname{lcm}(|a_1|,|a_2|)\neq 0$. Every element $an+x$ of $S(a,x)$ satisfies $an+x\equiv x\equiv b_i\pmod{a_i}$ (because $a_i\mid a$), so $S(a,x)\subseteq S(a_1,x)\cap S(a_2,x)=S(a_1,b_1)\cap S(a_2,b_2)$, with $x\in S(a,x)$. Hence the unions of members of $\mathcal{B}$ form a topology.

**2. Every nonempty open set is infinite.**
Each basic set $S(a,b)$ ($a\neq 0$) is infinite, being $\{\dots,b-a,b,b+a,b+2a,\dots\}$. A nonempty open set is a union of basic sets and therefore contains one, so it is infinite. Consequently **no nonempty finite set is open**; in particular $\{-1,1\}$ is not open.

**3. Each $S(a,b)$ is closed (indeed clopen).**
Take $a>0$. The $a$ progressions $S(a,b),S(a,b+1),\dots,S(a,b+a-1)$ are pairwise disjoint and cover $\mathbb{Z}$: by the division algorithm, every integer is congruent to exactly one of $b,b+1,\dots,b+a-1$ modulo $a$. Hence
$$\mathbb{Z}\setminus S(a,b)=\bigcup_{j=1}^{a-1} S(a,b+j),$$
a union of basic open sets, so the complement of $S(a,b)$ is open. (For $a=1$ this union is empty and $S(1,b)=\mathbb{Z}$.) Therefore $S(a,b)$ is closed as well as open.

**4. The non-units are a union over primes.**
An integer $n$ has a prime divisor iff $|n|\neq 1$: if $|n|\ge 2$ the least divisor of $|n|$ exceeding $1$ is prime, and $0$ is divisible by every prime, while $\pm 1$ are divisible by no prime. Thus
$$\mathbb{Z}\setminus\{-1,1\}=\bigcup_{p\ \text{prime}} S(p,0),\qquad S(p,0)=p\mathbb{Z}.$$

**5. Contradiction.**
By Step 3 each $S(p,0)$ is closed. If there were only finitely many primes, the right-hand side of the identity in Step 4 would be a finite union of closed sets, hence closed; so $\mathbb{Z}\setminus\{-1,1\}$ would be closed and its complement $\{-1,1\}$ would be open. But $\{-1,1\}$ is a nonempty finite set, which by Step 2 cannot be open. This contradiction shows the primes cannot be finite.

$$\boxed{\text{There are infinitely many primes.}\qquad\blacksquare}$$
