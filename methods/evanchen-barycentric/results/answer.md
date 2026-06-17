# Barycentric (Areal) Coordinates in Triangle Geometry

## The problem it solves

Olympiad triangle problems revolve around the triangle's special points (centroid, incenter, circumcenter, orthocenter, symmedian point) and around conditions of **collinearity, concurrence, perpendicularity, and concyclicity**. Synthetic chasing is brittle, and the external computational frames each break where you need them most: Cartesian gives hideous special-point coordinates and an unstructured circle equation; complex numbers handle circles but not the equation of a line (it needs conjugates); vectors have essentially no equation of a line at all. Barycentric coordinates fix every vertex and side on equal footing and turn the whole subject into linear/quadratic algebra adapted to the triangle.

## The key idea

Make the triangle its own coordinate frame. Write each point as a mass-weighted average of the vertices:
$$P = (x,y,z),\qquad P = xA+yB+zC,\qquad x+y+z=1.$$
Equivalently the normalized coordinates are ratios of signed sub-triangle areas (hence "areal"):
$$x=\frac{[PBC]}{[ABC]},\quad y=\frac{[PCA]}{[ABC]},\quad z=\frac{[PAB]}{[ABC]}.$$
Then $A=(1,0,0)$, $B=(0,1,0)$, $C=(0,0,1)$; side $BC$ is $x=0$, $CA$ is $y=0$, $AB$ is $z=0$; and a point dividing $BC$ in ratio $BX:XC=m:n$ is $X=(0:n:m)$.

## The formula set (with $a=BC,\ b=CA,\ c=AB$)

**Special points** (un-normalized $(u:v:w)$ denotes $\tfrac{1}{u+v+w}(u,v,w)$):
$$G=(1:1:1),\quad I=(a:b:c),\quad K=(a^2:b^2:c^2),\quad H=(\tan A:\tan B:\tan C),\quad O=(\sin 2A:\sin 2B:\sin 2C).$$

**Area** of $P_i=(x_i,y_i,z_i)$ (normalized):
$$[P_1P_2P_3]=[ABC]\cdot\det\begin{pmatrix}x_1&y_1&z_1\\x_2&y_2&z_2\\x_3&y_3&z_3\end{pmatrix}.$$

**Collinearity / line through two points** (scale-invariant, so usable un-normalized):
$$\det\begin{pmatrix}x_1&y_1&z_1\\x_2&y_2&z_2\\x_3&y_3&z_3\end{pmatrix}=0;\qquad \text{a line is } ux+vy+wz=0.$$
A line through $A$ has the form $vy+wz=0$; the three concurrent lines $u_ix+v_iy+w_iz=0$ concur iff $\det(u_i,v_i,w_i)=0$.

**Displacement vector** of normalized $P,Q$: $\vec{PQ}=(p_1-q_1,\,p_2-q_2,\,p_3-q_3)$, whose coordinates **sum to $0$**.

**Perpendicularity (EFFT).** For displacement vectors $(x_1,y_1,z_1)$ and $(x_2,y_2,z_2)$ (each summing to $0$),
$$\vec{MN}\perp\vec{PQ}\iff a^2(y_1z_2+y_2z_1)+b^2(z_1x_2+z_2x_1)+c^2(x_1y_2+x_2y_1)=0.$$
Specialization to a side: $\vec{PQ}=(x,y,z)\perp BC \iff a^2(z-y)+(c^2-b^2)x=0$; applying this to the displacement from $M=(0,\tfrac12,\tfrac12)$ gives the perpendicular bisector of $BC$ as $a^2(z-y)+(c^2-b^2)x=0$.

**Distance** of displacement $(x,y,z)$ (normalized; **not** scale-invariant):
$$|PQ|^2=-a^2yz-b^2zx-c^2xy.$$

**Circles.** Circumcircle $a^2yz+b^2zx+c^2xy=0$; general circle $-a^2yz-b^2zx-c^2xy+(ux+vy+wz)(x+y+z)=0$.

## Why the formulas are what they are

- The coordinates are translation-invariant precisely because $x+y+z=1$, so the position-vector origin may be moved to the **circumcenter $O$**, giving $A\cdot A=R^2$ and, via inscribed angle $\angle AOB=2C$ and the law of sines $c=2R\sin C$, $A\cdot B=R^2-\tfrac{c^2}{2}$ (cyclically).
- A displacement vector sums to $0$, so when EFFT/distance are expanded the entire $R^2$ contribution factors as $R^2(\sum x_1)(\sum x_2)=0$ and vanishes — leaving side-length-only formulas independent of $R$. This is exactly why area and distance need normalized representatives, while the homogeneous line/circle equations and scaled displacement-vector EFFT checks do not.

## The anchor problem, solved

**Claim.** $ABC$ acute scalene; $M,N,P$ midpoints of $BC,CA,AB$; the perpendicular bisectors of $AB,AC$ meet ray $AM$ at $D,E$; lines $BD,CE$ meet at $F$. Then $A,N,F,P$ are concyclic.

**Solution.** Set $A=(1,0,0),B=(0,1,0),C=(0,0,1)$, so $M=(0,\tfrac12,\tfrac12)$, $N=(\tfrac12,0,\tfrac12)$, $P=(\tfrac12,\tfrac12,0)$. Line $AM$ is $y=z$, so $D=(1-2t,t,t)$.

Impose $\vec{DP}\perp AB$ with $\vec{DP}=P-D=(2t-\tfrac12,\,\tfrac12-t,\,-t)$ and $\vec{AB}=(-1,1,0)$. EFFT gives
$$-a^2t+b^2t+c^2(3t-1)=0\ \Rightarrow\ t=j=\frac{c^2}{3c^2+b^2-a^2},\qquad D=(1-2j,j,j).$$
For $E=(1-2u,u,u)$, impose $\vec{EN}\perp AC$ with $\vec{EN}=(2u-\tfrac12,\,-u,\,\tfrac12-u)$ and $\vec{AC}=(-1,0,1)$. EFFT gives
$$-a^2u+b^2(3u-1)+c^2u=0\ \Rightarrow\ u=k=\frac{b^2}{3b^2+c^2-a^2},\qquad E=(1-2k,k,k).$$

Intersect $BD$ (through $B$, so $z/x=j/(1-2j)$) and $CE$ (through $C$, so $y/x=k/(1-2k)$). With $S=b^2+c^2-a^2$,
$$\frac rp=\frac{c^2}{S},\quad \frac qp=\frac{b^2}{S},\quad \frac1p=2+\frac{a^2}{S},\quad 2p-1=-\frac{a^2}{S}\,p,$$
so $F=(p,q,r)$ with $q=\tfrac{b^2}{S}p,\ r=\tfrac{c^2}{S}p$.

The homothety $X\mapsto 2X-A$ (center $A$, ratio $2$) fixes $A$ and sends $N\mapsto C$, $P\mapsto B$, so it carries the circle through $A,N,P$ to the circumcircle. Hence $A,N,F,P$ are concyclic **iff** $F'=2F-A=(2p-1,2q,2r)$ lies on the circumcircle $a^2yz+b^2zx+c^2xy=0$. Substituting,
$$a^2(2q)(2r)+b^2(2r)(2p-1)+c^2(2p-1)(2q)=\frac{p^2}{S^2}\big(4a^2b^2c^2-2a^2b^2c^2-2a^2b^2c^2\big)=0.$$
So $F'$ is on the circumcircle, $F$ is on the circle through $A,N,P$, and $A,N,F,P$ are concyclic. $\blacksquare$

## A short worked example (the formulas in action)

Take an explicit triangle to see the metric formulas compute real numbers. Let $a^2=4,\ b^2=5,\ c^2=6$ (a valid triangle, since each side-square is less than the sum of the other two).

- **Centroid** $G=(\tfrac13,\tfrac13,\tfrac13)$. **Distance $GA$:** the displacement $\vec{GA}=A-G=(\tfrac23,-\tfrac13,-\tfrac13)$ (sums to $0$), so
$$|GA|^2=-a^2(-\tfrac13)(-\tfrac13)-b^2(-\tfrac13)(\tfrac23)-c^2(\tfrac23)(-\tfrac13)=-\tfrac{4}{9}+\tfrac{10}{9}+\tfrac{12}{9}=2,$$
matching the median-length identity $m_a^2=\tfrac{2b^2+2c^2-a^2}{4}=\tfrac{10+12-4}{4}=\tfrac{18}{4}=\tfrac92$ with $|GA|=\tfrac23 m_a$, $|GA|^2=\tfrac49\cdot\tfrac92=2$.

- **Perpendicularity check:** using the displacement $A-M=(1,-\tfrac12,-\tfrac12)$, the side specialization $a^2(z-y)+(c^2-b^2)x$ gives $4(-\tfrac12+\tfrac12)+(6-5)(1)=1\neq0$, so the median is not perpendicular to $BC$, as expected. The foot of the altitude from $A$ is the point of $BC$ where this same expression *does* vanish — a one-line solve once the formula is in hand.

These are exactly the operations the anchor problem strings together: ratios for the midpoints, EFFT for the perpendicular bisectors, linear algebra for the intersections, and the circumcircle quadratic (after a homothety) for the concyclicity.
