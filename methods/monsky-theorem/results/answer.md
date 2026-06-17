# Monsky's Theorem

## The problem it solves

A square can be cut into $2,4,6,\dots$ triangles of equal area, but never into an **odd** number. Formally:

> **Theorem (Monsky).** If a square $S$ in the plane is dissected into $n$ triangles all of equal area, then $n$ is even. Equivalently, a square has no equidissection of odd cardinality.

The pieces may be arbitrary triangles meeting in any way (not necessarily edge-to-edge); the only hypothesis is that all areas are equal. The surprise is that a continuous, purely geometric act of cutting is obstructed by the *parity* of the count.

## The key idea

Parity of $n$ is a statement about divisibility by $2$, so measure everything with the **additive $2$-adic valuation** $v$, the function that reads off how many factors of $2$ a rational number carries. The single arithmetic fact that drives the proof:

$$v(n)=0 \iff n \text{ odd}, \qquad\text{so}\qquad v(1/n)=0 \text{ for odd } n.$$

Coordinates of a general dissection need not be rational, so $v$ is **extended from $\mathbb{Q}$ to the coordinate field of the dissection**; equivalently, one may extend it to all of $\mathbb{R}$ (Chevalley's extension theorem). Each relevant point $(x,y)$ is then **3-coloured** by comparing $v(x)$, $v(y)$, $v(1)=0$. Two facts collide:

- **(Area)** A triangle with one vertex of each colour has area whose valuation is $\le -1$ (the $-1$ comes from dividing by $2$ in the area formula), equivalently $|{\rm area}|_2\ge 2>1$. So its area cannot equal $1/n$ for odd $n$, where $v(1/n)=0$ and $|1/n|_2=1$.
- **(Existence)** Every dissection of the square contains such a tricoloured ("rainbow") triangle. This is the Sperner parity count applied to the dissection's basic segments; the **at-most-two-colours-on-a-line** lemma is what makes each side of each triangle behave like a two-colour boundary walk.

A rainbow triangle is therefore both *forced to exist* and *forbidden to have valuation $v(1/n)=0$* when $n$ is odd, a contradiction unless $n$ is even.

## The $2$-adic valuation, extended to $\mathbb{R}$

On $\mathbb{Q}$, write a nonzero rational uniquely as $x = 2^k\,(a/b)$ with $a,b$ odd; set $v(x)=k$, $v(0)=\infty$. Then

$$v(xy)=v(x)+v(y),\qquad v(x+y)\ge \min\{v(x),v(y)\},$$

with **equality $v(x+y)=v(x)$ whenever $v(x)<v(y)$** (a strictly dominant term survives). Here $v(1)=0$, $v(2)=1$, and for integers $v(n)\ge 1 \iff n$ even. With $|x|_2=2^{-v(x)}$, this means $|2|_2=\tfrac12$ and $|\tfrac12|_2=2$.

For a fixed finite dissection, let $F$ be the field generated over $\mathbb{Q}$ by all coordinates of the vertices and by the endpoints of side intersections used in the basic-segment graph. View $v$ through its **valuation ring** $\mathcal{O}=\{v\ge 0\}$; for the $2$-adic valuation on $\mathbb{Q}$ this is $\mathbb{Z}_{(2)}$ (fractions with odd denominator), a local ring with maximal ideal $(2)$.

> **Theorem (Chevalley).** For a field $K$, subring $R\subseteq K$, and prime $\mathfrak{p}\subseteq R$, there is a valuation ring $\mathcal{O}$ of $K$ with $R\subseteq\mathcal{O}$ and $\mathfrak{M}\cap R=\mathfrak{p}$.

Apply the prolongation form with $K_1=\mathbb{Q}$, $\mathcal{O}_1=\mathbb{Z}_{(2)}$, and $K_2=F$ (or $K_2=\mathbb{R}$). This gives a valuation ring on $F$ whose intersection with $\mathbb{Q}$ is $\mathbb{Z}_{(2)}$, so the associated additive valuation extends the original $v$ and keeps $v(2)=1$ and $v(\tfrac12)=-1$. (Proof sketch: Zorn's lemma on pairs (subring, proper ideal) gives a maximal local $(\mathcal{O},\mathfrak{M})$; maximality forces $\mathcal{O}$ to be a valuation ring -- if some $x$ had $x,x^{-1}\notin\mathcal{O}$, then $\mathfrak{M}$ would generate the unit ideal in both $\mathcal{O}[x]$ and $\mathcal{O}[x^{-1}]$, and combining two minimal finite relations lowers a degree, a contradiction.) The value group may enlarge from $\mathbb{Z}$ (for example $v(\sqrt2)=\tfrac12$), but the proof only compares valuations and uses the fixed rational values $v(1)=0$ and $v(\tfrac12)=-1$.

## The colouring

Colour every relevant point $(x,y)$ by comparing $v(x)$, $v(y)$, $v(1)=0$:

$$
\begin{aligned}
\textbf{A}:\;& v(x)>0 \text{ and } v(y)>0\\
\textbf{B}:\;& v(x)\le 0 \text{ and } v(x)\le v(y)\\
\textbf{C}:\;& v(y)\le 0 \text{ and } v(y)<v(x)
\end{aligned}
$$

The weak/strict asymmetry ($\le$ in B's cross-compare, $<$ in C's) makes A, B, C a **partition** of the plane: if both valuations are positive the point is A; if $v(x)\le0$, comparison of $v(x)$ and $v(y)$ puts it in exactly one of B or C; if $v(x)>0$ but $v(y)\le0$, it is C. The four corners of the unit square are coloured
$$
(0,0)\mapsto \textbf{A},\quad (1,0)\mapsto \textbf{B},\quad (0,1)\mapsto \textbf{C},\quad (1,1)\mapsto \textbf{B}.
$$

**Translation invariance.** Adding an A-point to a B-point leaves the x-coordinate valuation unchanged, since the B-point has $v(x)\le0$ and the A-point has positive x-valuation; the y-coordinate valuation remains at least that x-valuation, so the B-inequalities persist. For a C-point, the y-coordinate valuation is unchanged, and the new x-coordinate valuation is still strictly larger because both summands in that coordinate already have valuation larger than the C-point's y-coordinate. Thus the colouring is **translation-invariant under A-points** (and A is closed under negation). This is what lets the area formula be normalized.

## Lemma 1 (a line carries at most two colours)

**Statement.** Three points of three different colours are never collinear.

**Proof.** Collinearity of three points means the triangle they span has area $0$. But Lemma 2 below computes that *any* triple with one vertex of each colour — the computation uses only the colour conditions, not general position, so it covers the degenerate case — has area of valuation $\le -1$, hence $\ne 0$. So no line carries all three colours. $\blacksquare$

## Lemma 2 (rainbow triangle area)

**Statement.** A triangle with one vertex of each colour A, B, C has area $T$ with $v(T)\le -1$, equivalently $|T|_2\ge 2>1$.

**Proof.** By translation invariance, move the A-vertex to the origin (area and colours preserved). The vertices are $(0,0)\in\textbf{A}$, $(x_B,y_B)\in\textbf{B}$, $(x_C,y_C)\in\textbf{C}$, and
$$
T=\tfrac12\bigl(x_B y_C - x_C y_B\bigr).
$$
From B: $v(x_B)\le v(y_B)$ and $v(x_B)\le 0$. From C: $v(y_C)<v(x_C)$ and $v(y_C)\le 0$. Adding the two cross-inequalities,
$$
v(x_B)+v(y_C) \;<\; v(y_B)+v(x_C),\qquad\text{i.e.}\quad v(x_B y_C) < v(x_C y_B)\ \text{(strict)}.
$$
By the equality case of the ultrametric inequality, $v(x_B y_C - x_C y_B)=v(x_B y_C)=v(x_B)+v(y_C)\le 0$. Hence
$$
v(T)=v(\tfrac12)+v(x_B y_C - x_C y_B)\le -1 + 0 = -1. \qquad\blacksquare
$$

The $v(\tfrac12)=-1$ is exactly where the prime $2$ enters: dividing by $2$ lowers the additive valuation.

## Sperner's parity count for arbitrary dissections

For a dissection that is not necessarily edge-to-edge, first make a finite vertex set consisting of all triangle corners, all square corners, and every endpoint of an intersection between two sides of pieces. A T-junction contributes its point; an overlapping side segment contributes its two endpoints. A **face** is a side of one of the triangles or a side of the square, subdivided by these vertices. Two vertices on the same face are adjacent if no other such vertex lies between them on that face; the adjacent subsegments are **basic segments**. The boundary of every triangle and of the square is a union of basic segments.

Fix the colour pair A/B. A face whose endpoint colours are A and B contains an odd number of A/B basic segments: by Lemma 1 the whole face lies on a line with at most two colours, so the colour sequence along the face starts in A, ends in B, and switches A$\leftrightarrow$B an odd number of times. If a face's endpoints are not exactly A and B, then either it contains C and therefore cannot contain both A and B, or it contains only A/B and starts and ends with the same colour; in both cases it contains an even number of A/B basic segments.

If no original triangle had all three vertex colours, then each triangle would have either $0$ or $2$ A/B faces, so its boundary would contain an even number of A/B basic segments. Summing over all triangles modulo $2$, every interior basic segment is counted twice and cancels, so the square boundary would have an even number of A/B basic segments.

But on the square boundary, the bottom edge $y=0$ has $v(y)=\infty$, so it carries no C-point, only A and B, with endpoints A at $(0,0)$ and B at $(1,0)$; hence it contributes an odd number of A/B basic segments. The left edge $x=0$ has no B-point, and the right and top edges have no A-point, so they contribute none. The boundary contribution is odd, a contradiction. Therefore some **original** triangle in the dissection has vertices of all three colours.

## Theorem and proof

**Theorem (Monsky).** If a square is dissected into $n$ triangles of equal area, then $n$ is even.

**Proof.** Choose coordinates, translate/rotate the square, and scale it to $S=[0,1]^2$; this preserves the parity of the number of pieces and preserves equality of areas up to one common factor. Fix an extension of $v$ to the coordinate field of the dissection and colour the relevant vertices by A/B/C. Any dissection of $S$ into $n$ equal triangles, each of area $1/n$, contains a tricoloured original triangle by the Sperner parity count and Lemma 1. Its area $T$ satisfies $v(T)\le -1$ by Lemma 2. If $n$ were odd, then $v(n)=0$, so $v(1/n)=0$; but every piece, including the tricoloured one, has area $1/n$, forcing $v(T)=0$ -- contradicting $v(T)\le -1$. Therefore $n$ is even. $\blacksquare$

## Why it works — the mechanism in one line

Parity of $n$ is a $2$-divisibility fact, so the $2$-adic valuation, extended beyond rational coordinates, is the right lens; the determinant area formula turns the colour conditions into $v(\text{area})\le -1$, incompatible with $v(1/n)=0$ for odd $n$, while the forced A/B switch along the bottom edge gives a Sperner parity count, and the same determinant applied to a collinear triple keeps every line two-coloured, carrying the existence statement to arbitrary dissections.
