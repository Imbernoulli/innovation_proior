I have a sequence run by a quadratic recurrence: $x_0=1$, $x_1=x_2=a$, and $x_{n+1}=2x_nx_{n-1}-x_{n-2}$. I want to understand what happens when one of the terms is zero. "Periodic" is a strong global statement, and a single local fact — one term vanishes — is supposed to force it. So my first job is just to see the sequence, to get the shape of these numbers in front of me before I theorize.

Let me grind out the first terms as polynomials in $a$. I have $x_0=1$, $x_1=a$, $x_2=a$. Then $x_3=2x_2x_1-x_0=2a\cdot a-1=2a^2-1$. Next $x_4=2x_3x_2-x_1=2(2a^2-1)a-a=4a^3-2a-a=4a^3-3a$. And $x_5=2x_4x_3-x_2=2(4a^3-3a)(2a^2-1)-a$; let me expand $(4a^3-3a)(2a^2-1)=8a^5-4a^3-6a^3+3a=8a^5-10a^3+3a$, double it to $16a^5-20a^3+6a$, subtract $a$: $x_5=16a^5-20a^3+5a$.

Now I stare at $2a^2-1$, $4a^3-3a$, $16a^5-20a^3+5a$. These are not random polynomials. $2a^2-1$ and $4a^3-3a$ are exactly the expressions that appear when you write $\cos$ of a multiple angle in terms of $\cos$ of the angle: $\cos 2\theta=2\cos^2\theta-1$ and $\cos 3\theta=4\cos^3\theta-3\cos\theta$. And $16a^5-20a^3+5a$ is $\cos 5\theta$ in terms of $\cos\theta$. So if I set $a=\cos\theta$, then $x_3=\cos 2\theta$, $x_4=\cos 3\theta$, $x_5=\cos 5\theta$.

That is a very loud hint. Let me line up the angle multipliers: $x_3\mapsto 2$, $x_4\mapsto 3$, $x_5\mapsto 5$. And $x_0=1=\cos 0$, $x_1=a=\cos\theta$, $x_2=a=\cos\theta$. So the multipliers, indexed by $n=0,1,2,3,4,5$, run $0,1,1,2,3,5$. Those are the Fibonacci numbers. So I conjecture: writing $a=\cos b$, the term $x_n$ equals $\cos(F_n b)$, where $F_0=0$, $F_1=F_2=1$, and $F_{n+1}=F_n+F_{n-1}$.

This is too clean to be coincidence, but a guess from five data points isn't a proof. The recurrence is what I have to satisfy. So suppose, as an inductive hypothesis, that $x_k=\cos(F_k b)$ holds for all $k\le n$ (it does for $k=0,1,2$ by direct check). I need $x_{n+1}=2x_nx_{n-1}-x_{n-2}$ to come out as $\cos(F_{n+1}b)$. The product $x_nx_{n-1}=\cos(F_n b)\cos(F_{n-1}b)$, and there's a product-to-sum identity made exactly for this: $2\cos\alpha\cos\beta=\cos(\alpha+\beta)+\cos(\alpha-\beta)$. So

$$2x_nx_{n-1}=2\cos(F_n b)\cos(F_{n-1}b)=\cos\big((F_n+F_{n-1})b\big)+\cos\big((F_n-F_{n-1})b\big).$$

The first angle is $(F_n+F_{n-1})b=F_{n+1}b$ — the Fibonacci recurrence drops in exactly here. The second angle is $(F_n-F_{n-1})b$, and since $F_n-F_{n-1}=F_{n-2}$ (the recurrence read backwards), that is $\cos(F_{n-2}b)$. So $2x_nx_{n-1}=\cos(F_{n+1}b)+\cos(F_{n-2}b)$. But $x_{n-2}=\cos(F_{n-2}b)$ by the inductive hypothesis, so

$$x_{n+1}=2x_nx_{n-1}-x_{n-2}=\cos(F_{n+1}b)+\cos(F_{n-2}b)-\cos(F_{n-2}b)=\cos(F_{n+1}b).$$

The induction closes. The structure was hiding in plain sight: the doubling $2x_nx_{n-1}$ produces a sum of two cosines, one at the *sum* of consecutive Fibonacci indices and one at their *difference*, and the $-x_{n-2}$ is precisely tuned to cancel the difference term, leaving the sum term. The recurrence is the additive structure of the Fibonacci numbers transported through the cosine.

There's a catch I glossed over: I wrote $a=\cos b$, which only makes sense for a real angle $b$ when $|a|\le1$. If $|a|>1$ there is no real $b$ with $\cos b=a$. So I have to deal with $|a|>1$ before I lean on the cosine picture. Let me look at what the recurrence does when $|a|>1$. Intuitively, if the terms start bigger than $1$ in absolute value, the product $2x_nx_{n-1}$ should dominate and blow the sequence up, so it never gets near zero.

Let me try to prove $|x_{n+1}|\ge|x_n|$ for all $n$, which would say the magnitudes never decrease, in particular never drop to $0$. I'll carry along the stronger statement that the magnitudes are non-decreasing and all at least $1$. Base cases: $|x_0|=1$, and $|x_1|=|x_2|=|a|>1$, so indeed $|x_2|\ge|x_1|\ge|x_0|=1$. For the step, suppose $|x_k|\ge|x_{k-1}|\ge\cdots\ge|x_0|=1$ up to $k=n$. Then by the triangle inequality,

$$|x_{n+1}|=|2x_nx_{n-1}-x_{n-2}|\ge 2|x_n||x_{n-1}|-|x_{n-2}|.$$

My first instinct is to bound $-|x_{n-2}|$ using $|x_{n-2}|\le|x_{n-1}|$, which turns the right side into $2|x_n||x_{n-1}|-|x_{n-1}|=|x_{n-1}|(2|x_n|-1)\ge|x_{n-1}|$. But that lands on $|x_{n+1}|\ge|x_{n-1}|$, weaker than the $|x_{n+1}|\ge|x_n|$ I'm after — I factored out the wrong term. I want a clean factor of $|x_n|$ in front, so I should pair the lone $|x_{n-2}|$ against $|x_n|$ instead. Since $|x_{n-2}|\le|x_{n-1}|\le|x_n|$, I can bound $-|x_{n-2}|\ge-|x_n|$:

$$|x_{n+1}|\ge 2|x_n||x_{n-1}|-|x_{n-2}|\ge|x_n|(2|x_{n-1}|-1)\ge|x_n|,$$

where the middle step used $|x_{n-2}|\le|x_n|$. So $|x_{n+1}|\ge|x_n|\ge1$, the chain extends, and the induction holds for all $n$. Hence when $|a|>1$ every term has absolute value at least $1$ and no term is ever $0$.

So a zero can only occur when $|a|\le1$, and then a *real* $b\in[0,\pi]$ with $a=\cos b$ exists and $x_n=\cos(F_n b)$ for all $n$. I can also see the same obstruction through complex angles. If $a>1$, write $a=\cosh t$ and take $b=it$, so $\cos(F_n b)=\cosh(F_n t)$ and no term is zero. If $a<-1$, write $a=-\cosh t$ and take $b=\pi+it$; then $\cos(F_n b)=\cos(F_n\pi+iF_n t)=(-1)^{F_n}\cosh(F_n t)$, again never zero. The magnitude induction is the real-variable way to keep that case out, so from this point the zero case is the real-angle case.

Now I have the lever I need. Suppose $x_n=0$ for some index $n$. Note $n\ge1$, because $x_0=1\ne0$; and then $F_n\ge1$. From $x_n=\cos(F_n b)=0$, the angle $F_n b$ is where cosine vanishes, i.e. an odd multiple of $\tfrac{\pi}{2}$:

$$F_n b=\frac{k\pi}{2}\quad\text{for some odd integer }k.$$

The single zero pins $b$ to be a rational multiple of $\pi$. Solving, $b=\dfrac{k\pi}{2F_n}$. I want to write the whole sequence in terms of a clean rational-times-$2\pi$ angle, because the thing that will eventually repeat is a residue. Set $d=4F_n$ and $c=k$, both integers with $d\ge4$. Then

$$b=\frac{k\pi}{2F_n}=\frac{k}{4F_n}\cdot2\pi=\frac{c}{d}\,2\pi,$$

and I can check the zero is consistent: $F_n b=F_n\cdot\frac{c}{d}2\pi=F_n\cdot\frac{k}{4F_n}2\pi=\frac{k}{2}\pi$, an odd multiple of $\pi/2$, good. Now every term is

$$x_m=\cos(F_m b)=\cos\!\Big(\frac{F_m c}{d}\,2\pi\Big).$$

Because cosine has period $2\pi$, $x_m$ depends on $F_m c$ only through its residue modulo $d$, and with $c$ fixed it depends only on $F_m\bmod d$. So if I can show the Fibonacci residues $F_m\bmod d$ are eventually — indeed exactly — periodic in $m$, the sequence $x_m$ inherits that period.

Why are the Fibonacci numbers periodic modulo a fixed $d$? The honest reason is to look at consecutive pairs. Consider the pair $(F_m,F_{m+1})$ read in $\mathbb{Z}/d\mathbb{Z}$, an element of the finite set $(\mathbb{Z}/d\mathbb{Z})^2$, which has only $d^2$ elements. As $m$ ranges over $0,1,2,\dots$ the pairs cannot all be distinct, so there exist $n_1<n_2$ with

$$(F_{n_1},F_{n_1+1})\equiv(F_{n_2},F_{n_2+1})\pmod d.$$

I want this repetition to give a period that starts at the very beginning, not just eventually. The key is that the forward step $(u,v)\mapsto(v,u+v)$ on $(\mathbb{Z}/d\mathbb{Z})^2$ is invertible: from $(v,u+v)$ I recover $u=(u+v)-v$, so the inverse is $(v,w)\mapsto(w-v,v)$. A map that advances the pair and is a bijection on a finite set sends the sequence of pairs around a cycle, and a point on a cycle is revisited from index $0$ onward. Concretely, with $\ell=n_2-n_1$, apply the inverse step $n_1$ times to both sides of the congruence above: since the step is a bijection, $(F_{n_1},F_{n_1+1})\equiv(F_{n_2},F_{n_2+1})$ forces $(F_0,F_1)\equiv(F_\ell,F_{\ell+1})\pmod d$, and then advancing forward,

$$(F_m,F_{m+1})\equiv(F_{m+\ell},F_{m+\ell+1})\pmod d\quad\text{for all }m\ge0.$$

In particular $F_{m+\ell}\equiv F_m\pmod d$ for every $m\ge0$. This $\ell$ is a period of the Fibonacci sequence modulo $d$.

Now I cash this out on the $x$'s. For each $m$, $F_{m+\ell}\equiv F_m\pmod d$ means $d\mid(F_{m+\ell}-F_m)$, so $\dfrac{(F_{m+\ell}-F_m)c}{d}$ is an integer, hence $\dfrac{F_{m+\ell}c}{d}-\dfrac{F_m c}{d}\in\mathbb{Z}$. Adding an integer to the argument of $\cos(\,\cdot\,2\pi)$ changes nothing:

$$x_{m+\ell}=\cos\!\Big(\frac{F_{m+\ell}c}{d}2\pi\Big)=\cos\!\Big(\frac{F_m c}{d}2\pi+(\text{integer})\cdot2\pi\Big)=\cos\!\Big(\frac{F_m c}{d}2\pi\Big)=x_m.$$

So $x_{m+\ell}=x_m$ for all $m\ge0$: the sequence is periodic with period $\ell$.

The zero has done exactly one job: it turns the angle into a rational multiple of $2\pi$. The cosine formula turns the sequence into residues of Fibonacci numbers modulo $d$, and the invertible pair recurrence makes those residues repeat from the start. That gives one positive period $\ell$ for every term.
