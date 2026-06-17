Let me set up exactly what I am looking at. There is a top rail and a bottom rail. Starting from the left input terminals, the first thing the current meets is a resistor $R$ sitting in the top rail — a series resistor. Right after it, a resistor $R$ drops from the top rail straight down to the bottom rail — a shunt, a rung. Then another series $R$ in the top rail, another shunt $R$ to the bottom, and so on, the same cell over and over, marching off to the right with no end. The bottom rail is just plain wire. I drive a current in at the left terminals and I want the resistance the network presents there: the input resistance.

The obvious thing to try is brute series-and-parallel reduction from the far end inward, because that is how I collapse any ladder I can actually see the end of. Take a finite ladder of $n$ cells, find the resistance looking in, and let $n$ grow. So let me name the resistance looking into a ladder of $n$ cells as $X_n$. For one cell terminated by... terminated by what, though? That is already the snag. A finite ladder has to end in something — an open circuit, a short, a load — and the answer depends on that termination. The whole point of the problem is that there is no end. So let me at least watch the reduction and hope the dependence on the termination washes out.

Start from the rightmost cell of an $n$-cell ladder and fold inward. Whatever hangs to the right of a given shunt is some resistance $Y$. At that node the shunt $R$ is in parallel with $Y$, giving $RY/(R+Y)$, and then the series $R$ of the current cell sits in front of that, so looking in one more cell to the left the resistance is

$$Y \;\longmapsto\; R + \frac{R\,Y}{R+Y}.$$

So building the ladder up cell by cell is just iterating this one map. Call it $g(Y) = R + RY/(R+Y)$. Pick some termination $Y_0$ to seed it and apply $g$ over and over; $X_n = g^{(n)}(Y_0)$. Let me just turn the crank with $R=1$ and a couple of seeds to feel the behavior. Seed $Y_0 = 1$: $g(1) = 1 + 1/2 = 1.5$. Then $g(1.5) = 1 + 1.5/2.5 = 1.6$. Then $g(1.6) = 1 + 1.6/2.6 = 1.6154$. Then $1.6176$, then $1.6180$, and it has basically stopped moving. Try a wildly different seed, $Y_0 = 100$: $g(100) = 1 + 100/101 = 1.990$, then $g(1.99) = 1 + 1.99/2.99 = 1.6656$, then $1.6248$, $1.6190$, $1.6181$, $1.6180$. Same place. Seed $Y_0 = 0$ (a shorted end): $g(0) = 1$, then $1.5$, then $1.6$, and again $1.6180$. So whatever I terminate with, the iterates pile up on the same number near $1.618$, and they do it fast.

Good — that kills my worry about the termination, numerically. But "it seems to converge to about $1.618$ for $R=1$" is not an answer, and I have no closed form, and I do not love that I had to discover the existence of a limit by hand-cranking. There must be a reason the termination is forgotten, and the reason should hand me the number directly. Let me stare at why the seed gets forgotten.

The map quickly becomes a contraction on the physically relevant interval. Its derivative is $g'(Y) = R^2/(R+Y)^2$. For any nonnegative termination, the first application gives $R \le g(Y) < 2R$; once the iterates are in that interval, $|g'(Y)| \le R^2/(2R)^2 = 1/4$. So after the first fold, each additional cell cuts dependence on the arbitrary termination by at least a fixed factor. That is exactly the seed-independence I just saw. The finite ladders converge to a unique fixed point of $g$, the value $X$ with $g(X)=X$. But I want to actually understand *why the physics produces a fixed-point equation*, not just notice that the reduction map happens to have one. The contraction argument tells me the finite ladders converge; it does not by itself tell me what they converge to in terms of the circuit.

So let me go back to the circuit and think about what "no end" really buys me, because that is the feature I have not used yet — I have been treating the infinite ladder as a limit of finite ones, which is correct but indirect. The defining property of this network is that it is the *same cell repeated forever*. Let me take that literally, but I have to make the cut at the right place. Walk in from the input terminals through only the first series resistor $R$. I arrive at a node with two branches to the bottom rail: one branch is the first shunt resistor $R$, and the other branch goes right into the next series $R$, the next shunt $R$, the next series $R$, the next shunt $R$, and so on forever.

That right-hand branch, looked at from this node and the bottom rail, begins with a series $R$, then a shunt $R$, and then the same pattern forever. That is exactly the same description as the original input: a series $R$, a shunt $R$, and infinitely many identical cells after it. Removing one period from a semi-infinite periodic chain leaves another semi-infinite periodic chain of the same kind. Infinity minus one period is still the same infinity, so the network is self-similar and the right-hand remainder is a perfect copy of the whole ladder.

If the right-hand remainder is identical to the whole network, then it has the *same input resistance*. So let me name that resistance $X$ once and for all: $X$ is the input resistance of the semi-infinite ladder. Now look at the whole network as the first series $R$, followed by a node where the first shunt $R$ and the right-hand copy of the whole ladder both connect down to the bottom rail. By self-similarity, that copy has resistance $X$ as a one-port between this node and the bottom rail. So at the node just after the first series resistor, the first shunt $R$ is *in parallel with* a one-port of resistance $X$, and that parallel combination sits in series behind the first series $R$.

So I can write the input resistance of the whole thing in terms of itself:

$$X = R + \frac{R\,X}{R+X}.$$

The series $R$ contributes its $R$; the first shunt $R$ in parallel with the rest-of-ladder $X$ contributes $RX/(R+X)$; in series they add. And the $X$ on the right is the very same $X$ on the left, because the right-hand branch is the same semi-infinite ladder. This is exactly the fixed-point equation $g(X)=X$ that the contraction argument predicted, but now I see *why* it holds: it is not a property of an arbitrary iteration map, it is the statement that the network equals one cell in front of a copy of itself. The self-similarity is what closes the equation.

Now solve it. Multiply through by $(R+X)$ to clear the fraction:

$$X(R+X) = R(R+X) + R\,X.$$

Expand both sides:

$$RX + X^2 = R^2 + RX + RX.$$

The $RX$ on the left cancels one $RX$ on the right, leaving

$$X^2 = R^2 + RX,$$

i.e.

$$X^2 - R\,X - R^2 = 0.$$

A clean quadratic in $X$. The quadratic formula gives

$$X = \frac{R \pm \sqrt{R^2 + 4R^2}}{2} = \frac{R \pm \sqrt{5}\,R}{2} = R\,\frac{1 \pm \sqrt 5}{2}.$$

Two roots. The plus root is $R(1+\sqrt5)/2 \approx 1.618\,R$, positive. The minus root is $R(1-\sqrt5)/2 \approx -0.618\,R$, negative. A resistance built entirely from positive resistors, driven at its terminals, has to present a positive resistance — you cannot get a negative driving-point resistance out of a passive pile of resistors, the power dissipated would have to be negative. So the minus root is unphysical and I discard it. It is also exactly the spurious root the fixed-point iteration never approaches: every finite truncation $X_n$ was positive and the iterates climbed toward $1.618$, never toward $-0.618$. Both the physics and the convergence single out the same root. So

$$X = R\,\frac{1+\sqrt5}{2}.$$

That number $(1+\sqrt5)/2$ is the golden ratio $\varphi \approx 1.618$, and it dropped out with no input but the self-similarity of the ladder — and it lands exactly on the $1.618$ I watched the hand-cranked iteration converge to for $R=1$. Let me sanity-check the algebra against the defining identity of $\varphi$. The golden ratio satisfies $\varphi^2 = \varphi + 1$. My quadratic with $R=1$ is $X^2 = X + 1$, the very same equation — so $X=\varphi$ is forced, no arithmetic slip. And I can feed $\varphi$ straight back into the self-consistency equation as a final check: with $R=1$, $X = 1 + \varphi/(1+\varphi)$. Since $1+\varphi = \varphi^2$, the fraction is $\varphi/\varphi^2 = 1/\varphi = \varphi - 1$, so $X = 1 + (\varphi-1) = \varphi$. It reproduces itself exactly. The equation is consistent and the root is right.

So the whole thing collapses to a single causal chain: the ladder has no end, so after the first series resistor the branch continuing to the right is a copy of the whole; that copy has the same input resistance $X$, so the first shunt $R$ is in parallel with $X$ at that node; putting that parallel pair behind the first series resistor gives the self-consistency equation $X = R + RX/(R+X)$; clearing it gives $X^2 - RX - R^2 = 0$; the positive root, forced by passivity, is the input resistance

$$X = \frac{1+\sqrt5}{2}\,R = \varphi\,R.$$
