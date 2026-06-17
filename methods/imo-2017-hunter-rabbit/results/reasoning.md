Let me start with the part that feels most dangerous for the hunter. The rabbit and the hunter move at the same speed, so if the hunter ever knew the rabbit's exact position and the tracking reports stayed exact, the hunter could simply chase. The only real resource the rabbit has is uncertainty. A report is allowed to be wrong by one unit, and it arrives before the hunter moves. That one unit sounds small, but the hunter must make a visible move while the true side of that error is still unresolved.

The first instinct is to follow the reports, maybe not literally the last report but some average of many of them. If the rabbit feints left and right, a long average should reveal the forward drift. But this assumes there is a committed hidden path whose noise is merely being observed badly. The rules let the opposite picture matter: as long as several rabbit positions remain consistent with all the same reports, the hunter has to choose one move that works for all of them. If two legal histories give exactly the same reports and the hunter ends in the same place for both, then the rabbit only needs one of those histories to leave the hunter far away. So I should think about the set of possible rabbit positions, not a single point.

Call that set the cloud. If after some round the cloud is $C$, then before the next report the rabbit could be anywhere within distance $1$ of $C$; after a report $P$, the surviving cloud is the intersection of that one-step enlargement with the disk of radius $1$ centered at $P$. The report is legal whenever the intersection is nonempty. This is already enough to see why averaging can fail: the report can lie on a central line while two symmetric off-line histories remain alive, one above the line and one below it. The hunter sees the same data in both cases.

I want a quantitative version of that symmetric ambiguity. Suppose for a moment that the rabbit has even revealed its exact position. This only helps the hunter, so if the rabbit can still make progress from here, old information cannot save the hunter. Let the hunter be at $H$, the rabbit at $R$, and let $d=HR<100$. If $H\ne R$, choose the line $r$ through them, oriented from the hunter through the rabbit and onward; if $H=R$, choose any line $r$. I will try to make the rabbit spend a block of $200$ moves creating two possible endpoints, one unit above $r$ and one unit below $r$, while all reports stay on $r$.

Place two points $Y_1,Y_2$ symmetric about $r$, each at distance $1$ from $r$, each on the forward side of $R$, and each at distance $200$ from $R$. Let $Z$ be the midpoint of $Y_1Y_2$, so $Z$ lies on $r$. The segment from $R$ to $Y_i$ has length $200$ and never strays more than one unit from $r$, so the rabbit can move in $200$ unit steps along either segment. During the whole block, the tracking device can report the corresponding points on $r$; those reports are within distance $1$ of either possible rabbit path. Therefore the hunter receives exactly the same reports whether the rabbit is heading to $Y_1$ or to $Y_2$.

Let $R'$ be the point on $r$ with $RR'=200$. The foot $Z$ is slightly behind $R'$, because $RY_i=200$ and the perpendicular offset is $1$. Write
$$
\epsilon=R'Z=200-\sqrt{200^2-1}.
$$
This is positive and
$$
\epsilon=\frac{1}{200+\sqrt{200^2-1}},
$$
so $\frac1{400}<\epsilon<\frac1{200}$.
Also $(200-\epsilon)^2=200^2-1$, so
$$
\epsilon^2+1=400\epsilon.
$$

Now I need a lower bound on what the hunter can guarantee against these two candidates. The reports determine one final hunter point, say $Q$. If $Q$ is off $r$, one of $Y_1,Y_2$ is on the farther side of it, and projecting $Q$ to $r$ cannot increase the larger of the two distances. So the best case for the hunter may be searched on $r$. The block has only $200$ moves, hence the hunter's coordinate on $r$ cannot pass the point $H'$ with $HH'=200$. There is one tiny endpoint issue: if $Z$ lies before $H'$, the hunter could aim at $Z$ and leave distance exactly $1$ to both candidates. But that case means $d<\epsilon$, and since $\epsilon<1/200$, the bound $1>d^2+1/2$ is already more than enough. So I may now treat the main case $d\ge\epsilon$, where $Z$ is at or beyond $H'$ and the best reachable point on $r$ is $H'$ itself.

Since $H'R'=HR=d$, the signed distance from $H'$ to $Z$ is $d-\epsilon$. In the case now under consideration this is nonnegative, and in any event the square is the same. Therefore
$$
y^2=H'Y_1^2=H'Z^2+ZY_1^2=(d-\epsilon)^2+1
=d^2-2\epsilon d+\epsilon^2+1
=d^2+\epsilon(400-2d).
$$
As long as $d<100$, the factor $400-2d$ is greater than $200$, and with $\epsilon>1/400$ this gives
$$
y^2>d^2+\frac12.
$$
Together with the small case $d<\epsilon$, this says that in a block of $200$ rounds, from any revealed position with distance below $100$, the rabbit can choose legal reports and a legal path so that the final squared distance is more than the old squared distance plus $1/2$.

I should check the information step one more time. The hunter's moves during the block are determined by the reports. The two candidate paths give the same reports, so they force the same hunter endpoint. The larger of the distances from that endpoint to $Y_1,Y_2$ is at least the lower bound just computed: $1$ in the tiny case, and $y$ in the main case. The rabbit chooses the candidate realizing that larger distance. This does not require a false report, because the chosen candidate path was legal all along and every report stayed within one unit of it. The reveal at the start only removed information from the rabbit's advantage, so the same block argument is valid when the hunter knows less.

Now the accounting is straightforward. Starting from distance $0$, repeat this block while the distance is below $100$. Each block of $200$ rounds increases $d^2$ by more than $1/2$. After $2\cdot 100^2=20000$ blocks,
$$
200\cdot 20000=4\cdot 10^6
$$
rounds have passed, and the squared distance is greater than $100^2$. That is far less than $10^9$ rounds.

Once the distance is greater than $100$, the rabbit can keep it greater than $100$ until the required time: each round it moves one unit directly away from the hunter's current position, so before the hunter moves the distance has increased by one, and after the hunter's unit move the distance cannot be smaller than it was at the start of the round. Thus the hunter cannot guarantee being below $100$ after $10^9$ rounds. The answer is no.

So the constants are only serving one simple mechanism: the tracking error leaves a one-unit transverse ambiguity that the hunter cannot resolve before moving. A block converts that fixed lateral ambiguity into a fixed additive gain in $d^2$, not merely a temporary sideways error. Additive growth of the square means square-root growth of the distance, and that still escapes every fixed bound given enough rounds. Here the crude numbers already cross $100$ in under four million rounds, so a billion rounds gives the rabbit far more than it needs.
