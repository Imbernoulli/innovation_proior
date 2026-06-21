We are asked to bound the number of clubs in a town of $n$ citizens, where every club has odd size and every two distinct clubs share an even number of members, and to show that this number can be no more than $n$. The only rigid quantity in the problem is $n$, and it is attached to the ground set of citizens; the hypotheses say nothing about clubs being large or small, nested, or amenable to a greedy choice. They are pure parity facts. The natural first instinct is to count: sum the club sizes, or sum the pairwise intersection sizes. But every such attempt fails for the same reason. Ordinary integer counting sees the actual sizes $1,3,5,\ldots$ and the actual overlaps $0,2,4,\ldots$, while the hypotheses control those numbers only modulo $2$. Summing $\sum_i|S_i|$ gives a congruence with $m$ and no upper bound; summing $\sum_{i<j}|S_i\cap S_j|$ gives an even number that can be re-expressed through citizen degrees, and again the parity washes out. The counting sees too much. What is needed is a tool that retains exactly the parity information and nothing else, and then converts "at most $n$" into a statement about how much room there is to store that information.

I propose the dimension method. The idea is to stop counting and instead encode each club as a vector, choose the field that makes the hypotheses become clean algebra, and let the ambient dimension do the bounding. Encode each club $S$ by its indicator vector $v_S=(a_1,\ldots,a_n)$, where $a_x=1$ if citizen $x$ belongs to $S$ and $a_x=0$ otherwise. This loses no information about the club. The reason this encoding is the right one is that the two combinatorial quantities the problem talks about are already dot products: over the integers $\langle v_S,v_S\rangle=\sum_x a_x^2=\sum_x a_x=|S|$ since $a_x^2=a_x$ for a $0/1$ entry, and $\langle v_S,v_T\rangle=\sum_x a_x b_x=|S\cap T|$. So club size is a self-dot-product and intersection size is a cross-dot-product. The only mismatch with the hypotheses is that the problem does not care whether an overlap is $0$ or $2$ or $4$, only that it is even. That is precisely the cue to choose the field. Working over $\mathbb F_2$, the same indicator vectors live in $\mathbb F_2^n$ and the dot product is computed modulo $2$, so an odd number becomes $1$ and an even number becomes $0$. The two parity hypotheses therefore become
$$
\langle v_S,v_S\rangle=|S|\pmod 2=1,\qquad \langle v_S,v_T\rangle=|S\cap T|\pmod 2=0\quad(S\ne T).
$$
Listing the club vectors $v_1,\ldots,v_m$ and forming their Gram matrix $(\langle v_i,v_j\rangle)_{i,j}$, the diagonal entries are all $1$ and the off-diagonal entries are all $0$: the Gram matrix is the identity $I_m$ over $\mathbb F_2$.

What makes the method work is that this Gram structure forces the vectors to be linearly independent, and independence is exactly what the dimension $n$ caps. The independence proof has to be handled with care, because over $\mathbb F_2$ I cannot use the usual geometric reflex that a nonzero vector has positive squared length — for instance $(1,1)$ has self-dot-product $0$ in $\mathbb F_2^2$. So the argument must use only bilinearity, never positivity. Suppose there is a linear relation $\sum_i c_i v_i=0$ with coefficients $c_i\in\mathbb F_2$. Fix an index $j$ and dot the whole relation with $v_j$. By bilinearity,
$$
0=\Big\langle\sum_i c_i v_i,\,v_j\Big\rangle=\sum_i c_i\langle v_i,v_j\rangle=c_j,
$$
because every cross-term $\langle v_i,v_j\rangle$ with $i\ne j$ vanishes (even intersections) and the single surviving diagonal term $\langle v_j,v_j\rangle=1$ (odd size) carries the coefficient $c_j$. Since $j$ was arbitrary, every coefficient is zero, so the vectors are linearly independent. This is where both hypotheses pull their weight at once: even intersections kill the off-diagonal terms, and odd sizes keep the diagonal term alive, so dotting with $v_j$ reads off exactly the coefficient $c_j$. The argument is therefore not a counting argument at all but a coefficient-recovery argument. Once the vectors are independent, the bound is forced, because $\mathbb F_2^n$ has dimension $n$ and cannot hold more than $n$ independent vectors; hence $m\le n$. The bound is sharp: the $n$ singleton clubs have indicator vectors equal to the standard basis vectors $e_1,\ldots,e_n$, each with self-dot-product $1$ and pairwise dot products $0$, so the upper bound and the construction meet.

The same proof can be compressed into ranks, which makes the dimension count fully explicit. Place the club indicator vectors as the rows of an $m\times n$ incidence matrix $M$ over $\mathbb F_2$; then the $(i,j)$ entry of $MM^{\mathsf T}$ is $\langle v_i,v_j\rangle$, so the parity conditions give $MM^{\mathsf T}=I_m$, and since rank is submultiplicative and an $m\times n$ matrix has rank at most $n$, we get $m=\operatorname{rank}(I_m)=\operatorname{rank}(MM^{\mathsf T})\le\operatorname{rank}(M)\le n$. The family of clubs forces an $m$-dimensional identity to factor through an $n$-dimensional incidence space, so $m$ cannot exceed $n$. Finally, the choice of field is not cosmetic but load-bearing. Over $\mathbb R$ the Gram matrix would record the actual odd sizes $1,3,5,\ldots$ on the diagonal and the actual even overlaps $0,2,4,\ldots$ off the diagonal; the off-diagonal entries would not vanish merely because they are even, and the dot-with-$v_j$ calculation would leave unwanted cross-terms and fail to isolate $c_j$. Reducing modulo $2$ is exactly what turns the information I actually have, parity, into exact algebraic zeros and ones, so the independence argument closes and the dimension bound applies.

The finished result and its proof:

**Theorem (Oddtown).** In a town of $n$ citizens, suppose clubs are formed so that each club has an odd number of members and every two distinct clubs share an even number of members. Then the number of clubs is at most $n$, and this bound is sharp.

**Proof.** Let the clubs be $S_1,\ldots,S_m$, and let $v_i=v_{S_i}\in\mathbb F_2^n$ be the indicator vector of $S_i$, with a coordinate for each citizen. Over $\mathbb F_2$, dot products record parities:
$$
\langle v_S,v_S\rangle=|S|\pmod 2,\qquad \langle v_S,v_T\rangle=|S\cap T|\pmod 2.
$$
The hypotheses therefore give
$$
\langle v_i,v_i\rangle=1,\qquad \langle v_i,v_j\rangle=0\quad(i\ne j),
$$
so the Gram matrix $(\langle v_i,v_j\rangle)_{i,j}$ equals $I_m$ over $\mathbb F_2$.

Suppose $\sum_i c_i v_i=0$ with $c_i\in\mathbb F_2$. Dot both sides with $v_j$. By bilinearity,
$$
0=\Big\langle\sum_i c_i v_i,\,v_j\Big\rangle=\sum_i c_i\langle v_i,v_j\rangle=c_j.
$$
This holds for every $j$, so all $c_j=0$. The argument uses only bilinearity and the parity identities above, not positivity of a squared length, which is unavailable over $\mathbb F_2$. Hence $v_1,\ldots,v_m$ are linearly independent in the $n$-dimensional space $\mathbb F_2^n$, so $m\le n$.

The bound is attained by the $n$ singleton clubs, whose indicator vectors are the standard basis vectors of $\mathbb F_2^n$. $\qquad\blacksquare$

**Matrix form.** If $M$ is the $m\times n$ incidence matrix whose rows are the club vectors, then
$$
MM^{\mathsf T}=I_m,
$$
and therefore
$$
m=\operatorname{rank}(I_m)=\operatorname{rank}(MM^{\mathsf T})\le \operatorname{rank}(M)\le n.
$$
