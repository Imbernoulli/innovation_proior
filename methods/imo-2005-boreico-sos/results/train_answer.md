I will present the solution I would call Boreico's sign-isolating SOS method, as applied to IMO 2005 Problem 3. The problem asks us to show that for any positive real numbers x, y, z with xyz ≥ 1, the cyclic sum

(x⁵ − x²)/(x⁵ + y² + z²) + (y⁵ − y²)/(y⁵ + z² + x²) + (z⁵ − z²)/(z⁵ + x² + y²)

is nonnegative. At first glance this looks like it might be a routine fraction-sum inequality, but a closer inspection shows it is not. The numerator of the first term is x⁵ − x² = x²(x³ − 1), which is negative whenever x < 1. Nothing in the hypothesis prevents a variable from being small, because the other two variables can be chosen large enough to keep xyz ≥ 1. So the three summands are not individually nonnegative; the inequality is genuinely about a cancellation in which the positive contributions from variables above 1 must dominate the negative contributions from variables below 1. The expression is also not homogeneous, because scaling all three variables simultaneously changes the relative weights of the degree-five and degree-two terms, so we cannot simply normalize xyz to 1 by a global rescaling. Finally, the constraint is an inequality rather than an equality, which means we must use xyz ≥ 1 at some decisive step rather than substituting it once and forgetting about it.

The strategy I adopt is to bound each fraction below by a simpler fraction that all share the same denominator x² + y² + z². If I can find such bounds, then summing the three cyclic copies will be easy because the denominators match. The numerator I want for the first term is x² − yz, and the corresponding numerators for the other two terms are y² − zx and z² − xy by cyclic symmetry. The reason this choice is attractive is that

(x² − yz) + (y² − zx) + (z² − xy) = x² + y² + z² − (xy + yz + zx) = ½[(x − y)² + (y − z)² + (z − x)²],

which is a manifestly nonnegative sum of squares. So if I can prove the per-term inequality

(x⁵ − x²)/(x⁵ + y² + z²) ≥ (x² − yz)/(x² + y² + z²)

for each variable, then adding the three cyclic versions immediately gives

Σ_cyc (x⁵ − x²)/(x⁵ + y² + z²) ≥ ½[(x − y)² + (y − z)² + (z − x)²]/(x² + y² + z²) ≥ 0,

which is exactly the desired result.

The challenge is that the per-term inequality above is not true without the constraint xyz ≥ 1. If we take x = y = z = t with t small, then xyz = t³ < 1 and the left-hand side is negative while the right-hand side is zero, so the inequality fails. This tells me that the constraint must enter the proof of the per-term bound itself, and I would like to see exactly where it enters. The cleanest way to make that visible is to split the per-term bound into two steps.

First, I prove an unconditional algebraic inequality. I rewrite the right-hand side target through an intermediate fraction that has the same numerator x⁵ − x² as the original fraction but a different denominator. Specifically, observe that

(x² − 1/x)/(x² + y² + z²) = (x³ − 1)/[x(x² + y² + z²)] = x²(x³ − 1)/[x³(x² + y² + z²)] = (x⁵ − x²)/[x³(x² + y² + z²)].

So the intermediate fraction is (x⁵ − x²)/[x³(x² + y² + z²)]. Now compare the original fraction to this intermediate one. Their numerators are identical, so the comparison reduces to comparing the two denominators while keeping track of the sign of the common numerator. Instead of guessing, I compute the difference directly:

(x⁵ − x²)/(x⁵ + y² + z²) − (x⁵ − x²)/[x³(x² + y² + z²)] = (x⁵ − x²) · [x³(x² + y² + z²) − (x⁵ + y² + z²)] / [(x⁵ + y² + z²) · x³(x² + y² + z²)].

The bracket simplifies to (x³ − 1)(y² + z²), because x³(x² + y² + z²) = x⁵ + x³y² + x³z², and subtracting x⁵ + y² + z² leaves (x³ − 1)(y² + z²). Meanwhile x⁵ − x² = x²(x³ − 1). Therefore the whole numerator of the difference becomes

x²(x³ − 1) · (x³ − 1)(y² + z²) = x²(x³ − 1)²(y² + z²).

The factor (x³ − 1) appears twice and becomes a perfect square. Every remaining factor is nonnegative: x² ≥ 0, (x³ − 1)² ≥ 0, y² + z² > 0, and both denominators are positive. Hence the difference is nonnegative, and we have proved the unconditional step

(x⁵ − x²)/(x⁵ + y² + z²) ≥ (x⁵ − x²)/[x³(x² + y² + z²)] = (x² − 1/x)/(x² + y² + z²).

This step uses only algebra and does not invoke xyz ≥ 1 at all.

Second, I use the constraint to drop from (x² − 1/x)/(x² + y² + z²) to (x² − yz)/(x² + y² + z²). Since xyz ≥ 1 and x > 0, we have 1/x ≤ yz, and therefore x² − 1/x ≥ x² − yz. Dividing by the positive x² + y² + z² preserves the inequality, so

(x² − 1/x)/(x² + y² + z²) ≥ (x² − yz)/(x² + y² + z²).

This is the only place where xyz ≥ 1 is used, which makes the role of the constraint completely transparent. Equivalently, one can perform this last step by replacing the constant 1 with xyz in the numerator: x⁵ − x² ≥ x⁵ − x²·xyz = x³(x² − yz), and dividing by x³(x² + y² + z²) gives the same (x² − yz)/(x² + y² + z²). Either way, the direction is correct and the constraint is spent exactly once.

Chaining the two steps gives the per-term bound I wanted:

(x⁵ − x²)/(x⁵ + y² + z²) ≥ (x² − yz)/(x² + y² + z²).

Now I sum over the three cyclic permutations. Because every term on the right has the same denominator x² + y² + z², the sum telescopes beautifully:

Σ_cyc (x⁵ − x²)/(x⁵ + y² + z²) ≥ [(x² − yz) + (y² − zx) + (z² − xy)]/(x² + y² + z²) = ½[(x − y)² + (y − z)² + (z − x)²]/(x² + y² + z²) ≥ 0.

This completes the proof. Equality holds precisely when each of the nonnegative terms we used vanishes. The algebraic step requires (x³ − 1)²(y² + z²) = 0, and because y² + z² > 0, this forces x³ = 1, so x = 1. By cyclic symmetry we also get y = 1 and z = 1, and indeed xyz = 1 at this point. The final sum-of-squares step requires x = y = z, which is consistent. Thus equality occurs exactly at x = y = z = 1, and each original summand is zero there, as expected.

The name Boreico's sign-isolating SOS method captures the essence of what happened: the potentially negative terms are not ignored but isolated, routed through an intermediate fraction with the same numerator, and the resulting algebraic difference collapses to a sum of squares. The constraint xyz ≥ 1 is used surgically, only in the final drop from 1/x to yz, and the entire proof is driven by the common denominator x² + y² + z² that lets three cyclic lower bounds add up to the canonical sum-of-squares certificate.

To make the argument concrete, here is a small Python script that verifies the per-term inequality and the final sum inequality on a random sample of triples satisfying xyz ≥ 1.

```python
import random
import math

def per_term(x, y, z):
    return (x**5 - x**2) / (x**5 + y**2 + z**2)

def target(x, y, z):
    return (x**2 - y*z) / (x**2 + y**2 + z**2)

def verify_triple(x, y, z, tol=1e-9):
    lhs = per_term(x, y, z) + per_term(y, z, x) + per_term(z, x, y)
    rhs = 0.5 * ((x - y)**2 + (y - z)**2 + (z - x)**2) / (x**2 + y**2 + z**2)
    pt = per_term(x, y, z) - target(x, y, z)
    return lhs, rhs, pt

random.seed(0)
for _ in range(5000):
    x = 10 ** random.uniform(-2, 2)
    y = 10 ** random.uniform(-2, 2)
    min_z = 1.0 / (x * y)
    z = min_z * 10 ** random.uniform(0, 2)
    lhs, rhs, pt = verify_triple(x, y, z)
    assert lhs + 1e-9 >= rhs >= -1e-9, (x, y, z, lhs, rhs)
    assert pt + 1e-9 >= 0, (x, y, z, pt)

print("All 5000 random triples with xyz >= 1 satisfy the SOS bound.")
```
