# More-Sums-Than-Differences (MSTD) sets

## Problem

For a finite set of integers $A$, compare the sumset $A+A=\{a+a'\}$ and the difference set $A-A=\{a-a'\}$. Since addition commutes and subtraction does not, an unordered pair gives one sum but two differences, so the naive expectation is $|A-A|\ge|A+A|$. A set with $|A+A|>|A-A|$ is **sum-dominant** (an MSTD set). Goal: explicit infinite families of such sets, control of the imbalance $|A+A|-|A-A|$, and the proportion of subsets of $\{0,\dots,n-1\}$ that are sum-dominant.

## Key idea

The one asymmetry that helps is in the *opposite* direction from the naive count. The difference set is symmetric about $0$ ($c\in A-A\Rightarrow -c\in A-A$), so its missing values come in $\pm$ pairs — each structural omission costs **two** from the budget $2n-1$. The sumset has no such symmetry, so its omissions cost **one**. The contest is "who misses fewer values," and that pairing is the lever.

Two regimes exploit it:

1. **Explicit construction (local).** Start from a symmetric set, which is exactly balanced ($A+A=a^*+(A-A)$), and adjoin a single element that creates one fresh sum while every fresh difference is already present. The skeleton's balance is tipped by exactly one sum.

2. **Positive proportion (global).** For a uniform random subset, every sum/difference in the *middle* of the range has $\sim n/4$ representations and is present with overwhelming probability; only **fringe** values (near $0$, near $n-1$) can be missing. So pin the fringe to force a $\pm$-pair of missing differences against a single missing sum, and let the middle be random. A positive fraction of all $2^n$ subsets share any prescribed fringe, so a positive proportion are sum-dominant.

## The constructions, stated cleanly

**Seed.** $A_1=\{0,2,3,4,7,11,12,14\}$ has $A_1+A_1=[0,28]\setminus\{1,20,27\}$ and $A_1-A_1=[-14,14]\setminus\{\pm6,\pm13\}$, so $|A_1+A_1|=26>25=|A_1-A_1|$.

**Symmetric perturbed-AP family.** For $m\ge4$, $1\le d\le m-1$, $d\ne m/2$, $k\ge3$ ($d<m/2$) or $k\ge4$ ($d>m/2$):
$$B=[0,m-1]\setminus\{d\},\quad L=\{m-d,2m-d,\dots,km-d\},\quad a^*=(k+1)m-2d,$$
$$A^*=B\cup L\cup(a^*-B),\qquad A=A^*\cup\{m\}.$$
$A^*$ is symmetric about $a^*$ (balanced); $2m=m+m\in A+A$ but $2m\notin A^*+A^*$; and $A^*-\{m\}\subseteq A^*-A^*$ so $A-A=A^*-A^*$. Hence $|A+A|=|A^*+A^*|+1>|A-A|$. $(m,d,k)=(4,1,3)$ recovers $A_1$.

**Base-$b$ digit stacking (prescribed imbalance).** If $A$ is sum-dominant with $|A+A|=s$, $|A-A|=t$, then $A_N=A+bA+\dots+b^{N-1}A$ for large $b$ has $|A_N+A_N|=s^N$, $|A_N-A_N|=t^N$, so the imbalance compounds: $|A_N+A_N|=|A_N-A_N|^{\log s/\log t}$. Stacking shifted copies of $A_1$ at spacing $29>14$ realizes every integer imbalance: $S_{2k+1}=A_1+\{0,29,\dots,29k\}$ gives $|S+S|-|S-S|=2k+1$; deleting one interior point gives the even values; $\{0,\dots,|x|+1\}\cup\{2|x|+2\}$ gives negative $x$. The range of $|A+A|-|A-A|$ is all of $\mathbb{Z}$.

**Positive proportion.** Fix $L=\{0,2,3,7,8,9,10\}$ and $U=\{n-11,n-10,n-9,n-8,n-6,n-3,n-2,n-1\}$; let the middle be a uniform random subset of $\{11,\dots,n-12\}$. Then $U-L$ omits $n-7$, so $A-A$ misses $\pm(n-7)$ and $|A-A|\le 2n-3$. Meanwhile $L+L=\{0,\dots,20\}\setminus\{1\}$, $L+U=\{n-11,\dots,n+9\}$, $U+U=\{2n-22,\dots,2n-2\}$, and with probability $>1-6(2^{-7}+2^{-8})=119/128$ the random middle fills the rest, giving $A+A=\{0,\dots,2n-2\}\setminus\{1\}$ and $|A+A|=2n-2$. So $|A+A|=2n-2>2n-3\ge|A-A|$, and the count of sum-dominant subsets is $\ge 2^{n-22}\cdot\frac{119}{128}>(2\times10^{-7})2^n$ for $n\ge15$. A positive proportion are sum-dominant (and, by analogous fringes, positive proportions are difference-dominant and balanced).

**Average behavior.** Over all subsets, $\mathbb{E}|S+S|\sim 2n-11$ and $\mathbb{E}|S-S|\sim 2n-7$: on average $10$ sums and $6$ differences are missing, so differences win by $4$ on average — the commutativity instinct is correct as a mean statement, yet a positive proportion still buck it.

