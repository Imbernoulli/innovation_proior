The functional equation f(x+y)=f(x)+f(y) is deceptively simple. A quick check shows that every line f(x)=cx satisfies it, and elementary manipulations force f(0)=0, f(-x)=-f(x), and f(qx)=qf(x) for every rational q. In particular, once c=f(1) is fixed, the value of f is pinned on every rational number. The natural hope is that rational density then pushes the same formula to all real numbers, but that step fails: the equation only mentions addition, so it sees the real numbers only as a vector space over the rationals, not as a complete ordered field. That leaves enormous freedom. Any two rationally independent real numbers can be sent to independent values, and the freedom to do so is exactly what produces the wild "monster" solutions.

The correct resolution is Cauchy's functional equation with the Hamel-basis construction. View R as a vector space over Q. The Axiom of Choice, in the form of Zorn's Lemma, guarantees a Hamel basis H={e_alpha}: every real number has a unique finite representation as a rational linear combination of basis vectors. An additive function must be Q-linear, so it is determined entirely by its values on H. Choose any real values v_alpha for the basis vectors and extend Q-linearly. The resulting map is automatically additive. If the chosen values happen to be v_alpha=c e_alpha for a single constant c, the solution collapses back to the ordinary line f(x)=cx. Any other choice gives a pathological solution whose graph is dense in R^2, hence discontinuous everywhere, unbounded on every interval, non-monotone, and non-Lebesgue-measurable. The complete solution set is therefore the set of all Q-linear maps R -> R.

Any mild regularity hypothesis kills the monsters and restores uniqueness. If f is continuous at even a single point, additivity propagates continuity to every point, and rational approximation gives f(x)=cx. If f is bounded on any non-degenerate interval, the integer-scaling identity f(Nt)=Nf(t) imports that bound back to the origin and yields continuity at 0. If f is monotone, squeezing x between rationals forces f(x)=cx. And if f is merely Lebesgue measurable, Lusin's theorem supplies a large compact set on which f is uniformly continuous; a small shift of that set must overlap itself, producing two points separated by h on which f is controlled, which via additivity gives |f(h)|<epsilon and hence continuity at 0. Thus the only obstruction to f(x)=cx is the Choice-built freedom of a Hamel basis.

```python
import math
from fractions import Fraction

# A concrete finite-dimensional analogue: Q[sqrt(2)] inside R.
# Elements are represented as pairs (a, b) meaning a + b*sqrt(2) with rational a, b.
# This is what the full Hamel-basis construction looks like when the basis is just {1, sqrt(2)}.

SQRT2 = math.sqrt(2)

def qsqrt2_to_float(x):
    """Convert (a, b) with rational a,b to a real float."""
    a, b = x
    return float(a) + float(b) * SQRT2

# The additive equation forces f(q * x) = q * f(x) for rational q.
# On Q[sqrt(2)], every element is a*1 + b*sqrt(2), so f is determined by
# f(1) and f(sqrt(2)).  Linearity requires f(sqrt(2)) = sqrt(2) * f(1).
# Choosing any other value gives a "monster" on this subspace.

def make_monster(c, d):
    """
    Return an additive function on Q[sqrt(2)] with
        f(1) = c,   f(sqrt(2)) = d.
    If d != c*sqrt(2), this is not of the form f(x)=c*x.
    """
    def f(x):
        a, b = x  # a, b are Fractions
        return Fraction(c) * a + Fraction(d) * b
    return f

# Demonstrate additivity.
x = (Fraction(2, 3), Fraction(-1, 4))
y = (Fraction(5, 7), Fraction(3, 2))
f = make_monster(c=Fraction(3), d=Fraction(2015))
assert f((x[0]+y[0], x[1]+y[1])) == f(x) + f(y)

# Demonstrate rational homogeneity.
q = Fraction(11, 17)
assert f((q*x[0], q*x[1])) == q * f(x)

# This monster is not linear over R: compare f(sqrt(2))/sqrt(2) vs f(1)/1.
print("f(1) =", float(f((Fraction(1), Fraction(0)))))
print("f(sqrt(2)) =", float(f((Fraction(0), Fraction(1)))))
print("ratio at sqrt(2):", float(f((Fraction(0), Fraction(1)))) / SQRT2)
print("ratio at 1:", float(f((Fraction(1), Fraction(0)))) / 1.0)

# The graph is dense in the plane for a true monster over R.
# In the Q[sqrt(2)] analogue we can still see the analogous phenomenon:
# by varying rational coefficients q, r, the point (q + r*sqrt(2), q*c + r*d)
# traces a dense grid-like subset of the plane when c/d is irrational.
import random
random.seed(0)
points = []
for _ in range(1000):
    q = Fraction(random.randint(-100, 100), random.randint(1, 100))
    r = Fraction(random.randint(-100, 100), random.randint(1, 100))
    xv = float(q) + float(r) * SQRT2
    yv = float(q) * 3 + float(r) * 2015
    points.append((xv, yv))

# Regularity conditions force linearity on Q[sqrt(2)] as well.
# If f is bounded on any interval, the integer-scaling argument gives f=0.
# Here is the standard argument for boundedness on [-delta, delta].
def boundedness_implies_linear(c, d, M=10):
    """
    If |f(t)| <= M on Q[sqrt(2)] intersect [-delta, delta] for some delta>0,
    then f(Nt)=Nf(t) forces |f(t)| <= M/N whenever |Nt| <= delta.
    Taking N -> infinity shows f is continuous at 0, hence f(x)=c*x.
    """
    # Numerical illustration: choose t small; f(Nt) grows linearly in N,
    # so the only way to stay bounded is d = c*sqrt(2).
    t = (Fraction(0), Fraction(1, 1000))  # sqrt(2)/1000
    f = make_monster(c, d)
    values = [float(f((Fraction(n)*t[0], Fraction(n)*t[1]))) for n in range(1, 11)]
    return values

print("Values of monster along a small direction:", boundedness_implies_linear(Fraction(3), Fraction(2015)))
```
