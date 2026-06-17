# Semi-infinite resistor ladder by self-similarity

## The problem

A semi-infinite ladder is built from identical resistors $R$. Each repeating cell adds a series resistor $R$ in the top rail followed by a shunt resistor $R$ down to the bottom rail, and the cell repeats to the right without end. Driving the left input terminals, find the input resistance $X$.

## The key idea

The ladder is self-similar: from the node after the first series resistor, the branch continuing to the right begins again with a series $R$, then a shunt $R$, and continues forever, so that branch is identical to the original ladder and has the same input resistance $X$. At that node, the first shunt $R$ is therefore in parallel with a one-port of resistance $X$, and that parallel pair is in series with the first resistor.

## Derivation

The first series resistor $R$ is followed by a node where the shunt resistor $R$ and the right-hand remainder both connect to the bottom rail. By self-similarity the remainder has resistance $X$, so the shunt and the rest combine to $RX/(R+X)$, in series with the first $R$:

$$X = R + \frac{R\,X}{R+X}.$$

Multiply by $(R+X)$:

$$X(R+X) = R(R+X) + R\,X,$$
$$RX + X^2 = R^2 + RX + RX,$$
$$X^2 = R^2 + R\,X,$$
$$\boxed{X^2 - R\,X - R^2 = 0.}$$

By the quadratic formula,

$$X = \frac{R \pm \sqrt{R^2 + 4R^2}}{2} = R\,\frac{1 \pm \sqrt 5}{2}.$$

A passive resistor network must present a positive driving-point resistance, so the negative root $R(1-\sqrt5)/2 \approx -0.618\,R$ is discarded.

## Result

$$\boxed{\,X = \frac{1+\sqrt5}{2}\,R = \varphi\,R \approx 1.618\,R\,,}$$

the golden ratio $\varphi=(1+\sqrt5)/2$ times $R$.

**Check.** With $R=1$ the self-consistency equation is $X^2 = X + 1$, the defining identity of $\varphi$ ($\varphi^2=\varphi+1$), so $X=\varphi$. Substituting back, $X = 1 + \varphi/(1+\varphi) = 1 + \varphi/\varphi^2 = 1 + 1/\varphi = 1 + (\varphi-1) = \varphi$. For the finite-truncation reduction $g(Y)=R+RY/(R+Y)$, any nonnegative termination lands after one fold in $R \le g(Y) < 2R$, and on that interval $|g'(Y)|=R^2/(R+Y)^2 \le 1/4$; the finite ladders therefore converge to this same fixed point, confirming the semi-infinite limit is well-defined.
