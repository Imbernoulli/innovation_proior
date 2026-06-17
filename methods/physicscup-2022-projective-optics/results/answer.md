# Image of a point - projective construction

Four collinear, equally spaced points $A,B,C,D$ ($|AB|=|BC|=|CD|$) are imaged by a thin lens of unknown focal length, position and orientation; only the images $A',B',C'$ are given. The point $D'$ can be located by a ruler-and-compass construction, with no information about the lens, because the thin-lens imaging map is a **projective transformation** and the cross-ratio of four collinear points is a projective invariant.

## The three facts

**Lines map to lines, meeting on the lens.** If the object line $L=AB$ is parallel to the lens plane, the thin-lens equation gives one image distance and one fixed transverse magnification, so its image is a line. Otherwise, let a single light ray run along $L$. It refracts at the lens into an outgoing straight ray $L'$. Every object point on $L$ has its image on this outgoing ray or its straight extension, so the image of $L$ is the straight line $L'$. In the nonparallel case $L$ and $L'$ meet at the point where that ray crosses the lens; in the parallel case this is the corresponding point at infinity on the lens plane. Hence $A',B',C',D'$ are collinear, and the original line and its image meet on the lens in the projective sense.

**The map is a perspectivity.** For any point $P$, the ray through the optical center $O$ is undeviated, so $P$, $O$, $P'$ are collinear. Since $P'$ also lies on the image line $L'$, it is $OP\cap L'$ in the nondegenerate case. Thus the map $L\to L'$, $P\mapsto P'$, has all connecting lines $PP'$ through the single fixed point $O$: it is a perspectivity with center $O$, i.e. a projective transformation of the line.

**Cross-ratio is preserved.** For a pencil of four lines through $O$ cutting a transversal at $A,B,C,D$, the law of sines on the triangles at $O$ gives
$$(A,B;C,D)=\frac{CA}{CB}\cdot\frac{DB}{DA}=\frac{\sin\angle COA}{\sin\angle COB}\cdot\frac{\sin\angle DOB}{\sin\angle DOA},$$
which depends only on the directions at $O$, not on the transversal. So a perspectivity preserves the cross-ratio, and $(A',B';C',D')=(A,B;C,D)$.

## The invariant value

With $A,B,C,D$ equally spaced in order, take coordinates $A=0,B=1,C=2,D=3$:
$$(A,B;C,D)=\frac{CA}{CB}\cdot\frac{DB}{DA}=\frac{-2}{-1}\cdot\frac{-2}{-3}=2\cdot\frac23=\frac43.$$
Therefore $D'$ is the unique point on the line $A'B'C'$ with
$$(A',B';C',D')=\frac43.$$
The lens parameters never enter: any lens consistent with $A',B',C'$ gives the same $D'$.

## The geometric construction

Let $L'$ be the line through $A',B',C'$.
1. Pick a generic point $X$ off $L'$ so the auxiliary intersections are finite. On ray $A'X$ mark $Y,Z$ with $A'X=XY=YZ$, so $A',X,Y,Z$ are equally spaced and $(A',X;Y,Z)=\tfrac43$.
2. $P=(\text{line }B'X)\cap(\text{line }C'Y)$.
3. $D'=(\text{line }PZ)\cap L'$.

The point $P$ is the center of the perspectivity carrying the equally spaced range $A',X,Y,Z$ onto $A',B',C',D'$ (it fixes $A'$, sends $X\to B'$ and $Y\to C'$ by construction), so $(A',B';C',D')=(A',X;Y,Z)=\tfrac43$ - exactly the required condition. No numerical input is used; the construction is the cross-ratio condition realized with allowed geometric operations.

## Result

For the given data $A'=(1.166,1.180)$, $B'=(4.824,2.236)$, $C'=(6.310,2.666)$, imposing $(A',B';C',D')=\tfrac43$ along $L'$ yields
$$\boxed{D'=(7.115,\ 2.898)}.$$
The successive gaps $A'B'>B'C'>C'D'$ compress along $L'$ toward the vanishing point, as a perspective range must.
