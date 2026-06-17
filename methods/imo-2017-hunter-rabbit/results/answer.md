# Hunter-Rabbit Escape

No. The hunter cannot guarantee distance below $100$ after $10^9$ rounds.

Let the hunter be at $H$, the rabbit at $R$, and $d=HR<100$. Reveal $R$ to the hunter; this only helps the hunter. If $H\ne R$, let $r$ be the line through $H$ and $R$, oriented forward from $H$ through $R$; if $H=R$, choose any line $r$.

Choose two points $Y_1,Y_2$ symmetric about $r$, each one unit from $r$, each at distance $200$ from $R$. The rabbit can move in $200$ unit steps along either segment $RY_i$. Every point of either segment is within distance $1$ of $r$, so the tracking device may report points on $r$ throughout the block; the hunter sees the same reports for both possible rabbit paths.

Let $H'$ be the point on $r$ obtained by moving $200$ units forward from $H$. Let $Z$ be the midpoint of $Y_1Y_2$, and let $R'$ be the point on $r$ with $RR'=200$. Define
$$
\epsilon=R'Z=200-\sqrt{200^2-1}
=\frac{1}{200+\sqrt{200^2-1}},
$$
so $\frac1{400}<\epsilon<\frac1{200}$. If $d<\epsilon$, then the closest reachable point on $r$ to both candidates is $Z$, and even there the squared distance to either candidate is $1>d^2+\frac12$. Thus assume $d\ge\epsilon$. Then $Z$ lies at or beyond $H'$, and projection onto $r$ plus the $200$-step reachability bound show that the best final endpoint for the hunter against the two symmetric candidates is $H'$.

Since $(200-\epsilon)^2=200^2-1$, we have
$$
\epsilon^2+1=400\epsilon.
$$
Also $H'R'=d$, so $H'Z=d-\epsilon$ as a signed length. Thus the distance $y$ from $H'$ to either $Y_i$ satisfies
$$
y^2=(d-\epsilon)^2+1
=d^2-2\epsilon d+\epsilon^2+1
=d^2+\epsilon(400-2d).
$$
Because $d<100$, we have $400-2d>200$, and therefore
$$
y^2>d^2+\frac12.
$$

For any actual hunter endpoint, symmetry and projection onto $r$ show that the larger of the two distances to $Y_1,Y_2$ is at least the relevant lower bound above. Hence one of the two legal rabbit paths leaves the hunter at squared distance more than $d^2+\frac12$ after the $200$-round block.

Repeating while $d<100$, the squared distance increases by more than $1/2$ every $200$ rounds. After at most $2\cdot 100^2=20000$ blocks, the squared distance is greater than $100^2$, using at most
$$
200\cdot 20000=4\cdot 10^6<10^9
$$
rounds. Once the distance is greater than $100$, the rabbit keeps it greater than $100$ by moving directly away from the hunter each round; the hunter's unit move cannot reduce the distance below its previous value. Therefore no hunter strategy can guarantee the required bound.

Equivalently, the possible-position cloud can keep two symmetric candidates alive under one common sequence of reports. The hunter must move before that symmetry is resolved, and each block turns the one-unit lateral uncertainty into a fixed additive gain in the squared distance. This forces unbounded square-root growth of the distance.
