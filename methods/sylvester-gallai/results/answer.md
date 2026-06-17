# Sylvester–Gallai Theorem

## Statement

Let $S$ be a finite set of points in the Euclidean plane, not all lying on a single line. Then there exists a line that passes through **exactly two** points of $S$.

Such a line — one containing exactly two points of the set — is called an **ordinary line**. Equivalently: every finite non-collinear point set has an ordinary line.

## Setup and terminology

A **connecting line** of $S$ is a line containing at least two points of $S$. The theorem says that, when $S$ is not collinear, at least one connecting line is *not* "rich" (does not contain a third point), i.e. is ordinary.

For a point $P$ and a line $\ell$ with $P \notin \ell$, write $d(P,\ell)$ for the perpendicular distance from $P$ to $\ell$; this is a positive real number.

## Proof

**Choice of an extremal pair.** Consider all pairs $(P,\ell)$ such that $P \in S$, $\ell$ is a connecting line of $S$, and $P \notin \ell$.

- *At least one such pair exists.* Since $S$ is not collinear, take any two points of $S$; the line through them is a connecting line $\ell_0$, and because the points of $S$ are not all on $\ell_0$, some point $P_0 \in S$ lies off $\ell_0$. Then $(P_0,\ell_0)$ is such a pair.
- *There are only finitely many such pairs*, since $S$ is finite (finitely many points and finitely many connecting lines).

Hence the positive quantity $d(P,\ell)$ attains a minimum over these pairs. Fix a pair $(P,\ell)$ achieving it, and set $h := d(P,\ell) > 0$.

**Claim.** This $\ell$ is ordinary.

**Proof of claim (by contradiction).** Suppose $\ell$ contains at least three points of $S$.

Let $Q$ be the foot of the perpendicular from $P$ to $\ell$, so $PQ \perp \ell$ and $|PQ| = h$. The point $Q$ divides $\ell$ into two opposite closed rays (each containing $Q$). Among the $\ge 3$ points of $S$ on $\ell$, by the pigeonhole principle at least two lie on the **same** closed ray from $Q$; a point coinciding with $Q$ may be assigned to either ray. Call these two points $B$ and $C$, ordered along $\ell$ so that $B$ lies between $Q$ and $C$ (with $B$ possibly equal to $Q$). In particular $B \ne C$ and $B$ lies on the segment $QC$, so
$$|CB| \le |CQ|.$$

Let $m$ be the line through $P$ and $C$. Since $P, C \in S$ and $P \ne C$ (as $P \notin \ell$ while $C \in \ell$), $m$ is a connecting line.

*$B$ is off $m$.* If $B$ lay on $m$, then $B$, $C$, $P$ would be collinear; but $B$ and $C$ are distinct points of $\ell$, so the only line through both is $\ell$, forcing $P \in \ell$ — contradicting $P \notin \ell$. Hence $B \notin m$, so $(B,m)$ is a legitimate point–line pair with $d(B,m) > 0$.

*$d(B,m) < h$.* Let $B'$ be the foot of the perpendicular from $B$ to $m$, so $d(B,m) = |BB'|$.

First note that $B'$ lies on the segment $CP$. Put coordinates with $Q=(0,0)$, $\ell$ as the $x$-axis toward $C$, and $P=(0,h)$. Then $C=(c,0)$ for some $c>0$, and $B=(b,0)$ with $0\le b<c$. Points on the line $PC$ have the form $P+t(C-P)=(tc,h(1-t))$. The orthogonal projection of $B$ to this line has parameter
$$t=\frac{h^2+bc}{h^2+c^2},$$
so $0<t<1$. Thus $B'$ lies between $P$ and $C$.

Now consider the two right triangles
$$\triangle CQP \quad(\text{right angle at } Q),\qquad \triangle CB'B \quad(\text{right angle at } B').$$
They share the angle at $C$: because $B$ lies on segment $QC \subset \ell$, the ray $CB$ equals the ray $CQ$; and because $B'$ lies on segment $CP$, the ray $CB'$ equals the ray $CP$. Two right triangles sharing an acute angle are similar (AA), so $\triangle CQP \sim \triangle CB'B$ with the correspondence $C\leftrightarrow C,\ Q\leftrightarrow B',\ P\leftrightarrow B$. Therefore
$$\frac{|BB'|}{|PQ|} = \frac{|CB|}{|CP|}.$$
Now $|CB| \le |CQ|$ (shown above), and $|CQ| < |CP|$ because in the right triangle $CQP$ the side $CP$ is the hypotenuse while $CQ$ is a leg, and $h = |PQ| > 0$ makes the triangle non-degenerate. Hence $|CB| < |CP|$, the ratio is strictly less than $1$, and
$$d(B,m) = |BB'| = |PQ|\cdot\frac{|CB|}{|CP|} \;\le\; |PQ|\cdot\frac{|CQ|}{|CP|} \;<\; |PQ| = h.$$

So $(B,m)$ is an admissible pair with $d(B,m) < h$, contradicting the minimality of $(P,\ell)$.

The assumption that $\ell$ contains three or more points is therefore false: $\ell$ contains exactly two points of $S$, i.e. $\ell$ is ordinary. $\blacksquare$
