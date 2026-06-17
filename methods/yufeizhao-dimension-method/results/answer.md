# The Dimension Method for Oddtown

## Problem

In a town with $n$ citizens, every club has odd size and every two distinct clubs share an even number of members. The number of clubs is at most $n$, and the bound is sharp.

## Method

Encode each club $S$ by its indicator vector $v_S\in\mathbb F_2^n$, one coordinate per citizen. Over $\mathbb F_2$, dot products record parities:
$$
\langle v_S,v_S\rangle=|S|\pmod 2,\qquad
\langle v_S,v_T\rangle=|S\cap T|\pmod 2.
$$
Thus odd club sizes give diagonal entries $1$, and even pairwise intersections give off-diagonal entries $0$.

## Proof

Let the clubs be $S_1,\ldots,S_m$, and let $v_i=v_{S_i}\in\mathbb F_2^n$. The hypotheses give
$$
\langle v_i,v_i\rangle=1,\qquad \langle v_i,v_j\rangle=0\quad(i\ne j).
$$
Equivalently, the Gram matrix $(\langle v_i,v_j\rangle)_{i,j}$ is $I_m$ over $\mathbb F_2$.

Suppose $\sum_i c_i v_i=0$ with $c_i\in\mathbb F_2$. Dot both sides with $v_j$. By bilinearity,
$$
0=\left\langle\sum_i c_i v_i,v_j\right\rangle=\sum_i c_i\langle v_i,v_j\rangle=c_j.
$$
This holds for every $j$, so all $c_j=0$. This uses only bilinearity and the parity identities above, not positivity of a squared length, which is unavailable over $\mathbb F_2$. The vectors $v_1,\ldots,v_m$ are linearly independent in the $n$-dimensional space $\mathbb F_2^n$, hence $m\le n$.

The bound is attained by the $n$ singleton clubs. Their indicator vectors are the standard basis vectors of $\mathbb F_2^n$.

## Matrix Form

If $M$ is the $m\times n$ incidence matrix whose rows are the club vectors, then
$$
MM^{\mathsf T}=I_m.
$$
Therefore
$$
m=\operatorname{rank}(I_m)=\operatorname{rank}(MM^{\mathsf T})\le \operatorname{rank}(M)\le n.
$$

## Why the Field Matters

Over $\mathbb R$, the Gram matrix records the actual odd sizes and actual even overlap sizes, so the off-diagonal entries need not be zero. Reducing modulo $2$ turns exactly the available information into algebraic structure: odd becomes $1$, even becomes $0$, and the Gram matrix becomes the identity. The dimension bound follows from linear independence, not from positivity or Euclidean length.
