The canonical name for the method I am presenting is the Kalda infinite ladder technique, also called the infinite-ladder self-similarity method. It is the standard way to find the input resistance of a semi-infinite ladder made of identical resistors, and the key move is to exploit the exact self-similarity of the network rather than treating it as a limiting finite circuit.

Here is the situation. The ladder is built from identical resistors of value R. Starting from the left input terminals, the first thing the current meets is a resistor R in the top rail, a series resistor. Immediately after that, a resistor R drops from the top rail down to the bottom rail, forming the first shunt or rung. Then the pattern repeats: another series R in the top rail, another shunt R down to the bottom rail, and so on forever. The bottom rail is just a continuous wire. We want the resistance seen looking in from the left terminals.

The obvious first thought is to compute the resistance of a finite ladder with some termination at the far end and then let the ladder grow without bound. But a finite ladder has to end in something, and the result depends on that ending. An open circuit, a short circuit, or any finite load all give different input resistances for a finite ladder. The whole point of the semi-infinite ladder is that there is no end at all, so the finite truncation approach leaves an annoying dependence on an arbitrary choice. The Kalda method removes that dependence by noticing that the infinite ladder is identical to itself after one cell.

To see the self-similarity, imagine walking in from the input terminals through only the first series resistor. I arrive at the node just after it. From that node, two paths reach the bottom rail. One path is the first shunt resistor R. The other path continues to the right through another series R, then another shunt R, then another series R, then another shunt R, and never stops. But that continuation is exactly the same kind of semi-infinite ladder I started with. Removing one period from an infinite periodic structure leaves the same infinite periodic structure. So the branch continuing to the right is a perfect copy of the whole ladder, just shifted by one cell.

Because the right-hand branch is identical to the whole ladder, it must present the same input resistance. Let me call that resistance X. Then the whole ladder can be redrawn as the first series R, followed by a node where the first shunt R and a one-port of resistance X both connect to the bottom rail. The shunt R and the right-hand remainder X are in parallel, and that parallel combination is in series with the first R. Therefore the input resistance X satisfies the self-consistency equation

X = R + (R X)/(R + X).

The term R is the first series resistor. The term (R X)/(R + X) is the parallel combination of the first shunt R and the rest-of-ladder X. The fact that the same X appears on both sides is exactly the statement of self-similarity. This equation is also the fixed-point equation of the finite-truncation reduction map, but here it arises directly from the circuit topology rather than from an iterative numerical procedure.

To solve it, multiply both sides by (R + X) to clear the denominator:

X(R + X) = R(R + X) + R X.

Expanding gives

R X + X^2 = R^2 + R X + R X.

The R X term on the left cancels one of the R X terms on the right, leaving

X^2 = R^2 + R X,

or equivalently

X^2 - R X - R^2 = 0.

This is a quadratic equation in X. Applying the quadratic formula,

X = (R ± sqrt(R^2 + 4 R^2))/2 = R (1 ± sqrt(5))/2.

There are two roots. The positive root is R(1 + sqrt(5))/2, approximately 1.618 R. The negative root is R(1 - sqrt(5))/2, approximately -0.618 R. Since the ladder is built entirely from positive resistors, it must present a positive driving-point resistance; a negative resistance would correspond to negative power dissipation, which is impossible for a passive network. So the negative root is unphysical and is discarded. The positive root is the golden ratio φ = (1 + sqrt(5))/2 times R. Thus the input resistance is

X = φ R ≈ 1.618 R.

This result has several quick consistency checks. With R = 1, the self-consistency equation becomes X^2 = X + 1, which is precisely the defining identity of the golden ratio, φ^2 = φ + 1. Substituting X = φ back into the original equation with R = 1 gives φ = 1 + φ/(1 + φ). Since 1 + φ = φ^2, the fraction simplifies to φ/φ^2 = 1/φ = φ - 1, so the right-hand side is 1 + (φ - 1) = φ, confirming the equation holds. Another check comes from the finite-truncation reduction map g(Y) = R + R Y/(R + Y). For any nonnegative termination, one application lands the value in the interval [R, 2R), and on that interval the derivative satisfies |g'(Y)| = R^2/(R + Y)^2 ≤ 1/4. Therefore the map is a contraction, and every finite truncation converges to the same fixed point regardless of its termination. That fixed point is exactly the X we found, so the semi-infinite limit is well defined and equal to φ R.

The Python code below verifies the result numerically. It defines the finite-ladder reduction map, iterates it from a few different terminations, and compares the limit with the closed-form golden-ratio expression.

```python
import math

def reduce_ladder(Y, R=1.0):
    """One-cell reduction: series R in front of parallel(R, Y)."""
    return R + (R * Y) / (R + Y)

def finite_ladder_input(n, termination, R=1.0):
    """Fold in n cells from the right with the given termination."""
    Y = termination
    for _ in range(n):
        Y = reduce_ladder(Y, R)
    return Y

R = 1.0
phi = (1.0 + math.sqrt(5.0)) / 2.0
closed_form = phi * R

print(f"Closed-form input resistance: {closed_form:.12f} * R")
print("Finite-ladder convergence from several terminations:")
for termination in [0.0, 1.0, 100.0]:
    approx = finite_ladder_input(30, termination, R)
    print(f"  termination={termination:>6}: X_30 = {approx:.12f}, error = {abs(approx - closed_form):.2e}")

# Direct self-consistency check.
X = closed_form
self_consistent = R + (R * X) / (R + X)
print(f"Self-consistency X = R + R*X/(R+X): {self_consistent:.12f}")
assert math.isclose(self_consistent, closed_form, rel_tol=1e-12)
```

In summary, the Kalda infinite ladder technique replaces the awkward limit of finite ladders with a single self-similarity argument. The infinite ladder equals one cell followed by a copy of itself, so its input resistance satisfies a quadratic equation whose physically admissible root is the golden ratio times the unit resistance. The finite-truncation map converges to that same value from every nonnegative termination, confirming that the semi-infinite answer is both unique and robust.
