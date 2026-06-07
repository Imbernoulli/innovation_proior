I am in the library reading John Burr Williams, *The Theory of Investment Value*, and I want to turn his idea into a rule an investor could actually follow. Williams says the value of a stock is the present value of its future dividend stream — discount each future dividend back to today and add them up. That is a valuation of a single security. But I want a rule for the thing the investor actually decides: not the value of one stock, but how to split money across many of them. So let me try to push Williams' rule up from one security to a whole portfolio and see what it tells me to do.

The honest version of Williams under uncertainty is this: dividends are not known, so the value of a security is the *expected* value of its discounted future dividend stream. Call that Rᵢ for security i — the anticipated discounted return per dollar in security i. If I put a fraction Xᵢ of my money in security i, with Xᵢ ≥ 0 (I am not going to allow short sales for now) and Σ Xᵢ = 1 (all the money is invested), then the portfolio's anticipated return is R = Σ Xᵢ Rᵢ. Williams' rule, lifted to the portfolio, says: choose the Xᵢ to maximize R. So maximize Σ Xᵢ Rᵢ subject to Xᵢ ≥ 0, Σ Xᵢ = 1.

Let me just look at what that maximization actually wants. R = Σ Xᵢ Rᵢ is a weighted average of the numbers Rᵢ, where the weights Xᵢ are nonnegative and sum to one. A weighted average of fixed numbers can never exceed the largest of those numbers, and it equals the largest exactly when all the weight sits on the largest one. So to maximize R I set Xᵢ = 1 for whichever security has the biggest Rᵢ, and zero for everything else. If two or three securities tie for the maximum, any split among *those* is equally good — but it never *helps* to split, and it never helps to touch any security that isn't tied for the top. The rule says: put everything in one security.

That stops me cold. Everybody diversifies. Spreading your money across several securities is the single most ordinary, sensible thing an investor does — it is so commonplace it is a proverb, don't put all your eggs in one basket. And here is the most faithful reading of the best valuation theory I have, and it tells me to do the exact opposite: pile everything onto the one security with the highest expected return. A rule that never, for any beliefs about the securities, prefers a diversified portfolio to all the undiversified ones cannot be the rule rational investors follow, and it cannot be the rule I would advise them to follow. It has to be rejected — not as an approximation that's a bit off, but as descriptively and normatively wrong about the one behavior we are most sure about.

And notice it doesn't matter how I formed the Rᵢ. Different discount rates for different securities, a risk allowance folded into the anticipated return the way Hicks does it for a firm, discount rates that drift over time — none of that changes the shape of the argument. As long as the rule is "maximize a single anticipated-return number per security, summed with the weights," the objective is a weighted average and the maximizer is a corner. So I cannot rescue this by being cleverer about the discount rate. The defect is in the *form* of the rule: one number per security, maximize the weighted sum. That form has no room for diversification in it.

So why *do* people diversify? Not to raise their expected return — we just saw concentration does that. They diversify to reduce uncertainty. They are afraid of the spread of outcomes, not just the center. If the future were known with certainty, an investor really would put everything in the single highest-return security; there'd be no reason to do anything else. It is precisely *because* returns are uncertain that spreading is sensible. So uncertainty is not a nuisance to be smoothed over — it is the whole reason the actual behavior exists. Which means my rule has thrown away exactly the thing that generates the behavior I'm trying to explain. The expected return Rᵢ is the *center* of security i's distribution of outcomes; by keeping only the center I deleted the spread, and the spread is what the investor cares about when they diversify.

Let me try the obvious patch before I do anything more drastic, because maybe diversification can be bolted onto the expected-return rule after all. The patch everyone reaches for: among the securities with the highest expected return, spread your money, and lean on the law of large numbers — average enough things together and the realized yield of the portfolio gets close to its expected yield, so you get the high mean *and* the safety of averaging. If that worked, there'd be a single portfolio that simultaneously has maximum expected return and, by the averaging, minimum dispersion, and I'd just recommend that and be done.

But it doesn't work, and the reason it doesn't work is the same reason it's tempting. The law of large numbers is a statement about *independent* (or nearly independent) draws: average many independent random variables and the average concentrates on its mean. Security returns are nothing like independent. When the market falls, most stocks fall together; firms in the same industry do badly at the same time. The returns are heavily intercorrelated. So averaging them does not make the dispersion vanish — there is a floor below which spreading cannot push the portfolio's variability, set by how much the securities move together. Diversification cannot eliminate all the dispersion. And the premise of the patch is false too: the maximum-expected-return portfolio is generally *not* the minimum-dispersion portfolio. They are different portfolios. You can gain expected return by accepting more dispersion, or cut dispersion by giving up expected return — there is a *rate* at which you trade one for the other, not a single point that is best on both counts. So I cannot get diversification for free by averaging. If I want a rule that recommends diversification, dispersion has to be in the objective itself, as a thing the investor is willing to pay expected return to reduce.

So there are two things the investor cares about, pulling against each other: the expected return of the portfolio, which they want high, and the dispersion of the portfolio's return, which they want low. I need a number for the dispersion. The natural one from statistics is the variance — the average squared deviation of the return from its mean — or equivalently its square root, the standard deviation. Let me adopt variance, write it as V, and see whether putting V into the picture finally produces diversification.

Whether variance is the *right* dispersion measure comes down to the same test the expected-return rule failed: does the variance of a *portfolio* reward spreading? The portfolio return is the weighted sum R = Σ Xᵢ Rᵢ, with the Xᵢ chosen by me and the Rᵢ random. So I need the variance of a weighted sum of random variables. And the variance of a weighted sum is emphatically *not* the weighted sum of the variances. Let me write it out. With E for expected return,

    E = Σᵢ Xᵢ μᵢ,        μᵢ = E(Rᵢ),

and for the variance, expanding the square of the deviation,

    V = E[(Σᵢ Xᵢ (Rᵢ − μᵢ))²] = Σᵢ Σⱼ Xᵢ Xⱼ E[(Rᵢ − μᵢ)(Rⱼ − μⱼ)] = Σᵢ Σⱼ Xᵢ Xⱼ σᵢⱼ,

where σᵢⱼ = E[(Rᵢ − μᵢ)(Rⱼ − μⱼ)] is the covariance of securities i and j, and σᵢᵢ is just the variance of security i. The diagonal terms Σᵢ Xᵢ² σᵢᵢ are the "weighted sum of variances" I'd have naively guessed; but there is a whole second batch, the off-diagonal Σᵢ≠ⱼ Xᵢ Xⱼ σᵢⱼ, the covariances. Writing the covariance through the correlation, σᵢⱼ = ρᵢⱼ σᵢ σⱼ, those cross terms carry ρᵢⱼ.

This is the thing I was missing, and it is exactly what makes variance the right measure. Stare at the cross terms. If two securities are not perfectly correlated, ρᵢⱼ < 1, then their covariance σᵢⱼ = ρᵢⱼ σᵢ σⱼ is smaller than σᵢ σⱼ, and the variance of the mix comes out *below* what you'd get from the variances alone. Take the simplest case, half in each of two securities with equal variance σ²: V = ¼σ² + ¼σ² + 2·¼·ρσ² = ½σ²(1 + ρ). If ρ = 1 (the returns move in lockstep) this is σ², no improvement — spreading across two identical-behaving things does nothing. But for any ρ < 1 it is strictly less than σ², and at ρ = 0 it is ½σ², and for ρ < 0 it is smaller still. So mixing imperfectly correlated securities *reduces* variance, and the less correlated they are, the more it reduces. The covariances, the very terms that made the variance formula more complicated than a weighted sum, are the mechanism of diversification. Variance is the right risk measure precisely *because* it is a weighted sum that drags in all the covariances — diversification is built into it, automatically.

And this immediately tells me what *kind* of diversification is the right kind, which the proverb never could. It is not about holding *many* securities. Sixty railway stocks are barely diversified, because railway returns move together — high covariances among themselves, so the cross terms stay large and the variance barely drops. The same number of names spread across railroads, utilities, mining, several manufacturers cuts the variance much more, because firms in different industries, with different economic drivers, have lower covariances. So "diversify" sharpens into "hold securities with low covariances among themselves," and the variance formula is what says so. The proverb was right to spread the eggs but silent on which baskets; the covariance terms name the baskets.

Now I have two criteria, E up, V down, and they trade off — so what is the rule? There is no single best portfolio, because the investor who wants more E must accept more V, and how much more is a matter of their taste, which I don't know and don't want to assume. What I *can* do, as a matter of pure logic before any taste enters, is throw out the portfolios that are beaten on both counts. A portfolio is worth offering only if nothing else gives at least as much E with less V, or at least as little V with more E. Call those the efficient portfolios — minimum V for a given E (or higher E), maximum E for a given V (or lower V). Every sensible investor, whatever their exact appetite for risk, will choose one of these; the rest are dominated and can be discarded. So the rule is not "compute the one optimal portfolio," it is "compute the efficient set, hand the investor that curve, and let them pick the point on it that matches their willingness to trade variance for return." The set of attainable (E, V) pairs is some region; its upper-left boundary — high E, low V — is the efficient frontier, and the answer to the whole problem is that frontier.

I should make sure this efficient set is something I can actually find and draw, not just define. Let me take the smallest case that already shows the geometry: three securities. The constraint Σ Xᵢ = 1 means I don't have three free weights, only two — set X₃ = 1 − X₁ − X₂ and work in the plane of (X₁, X₂). The no-short-sale conditions X₁ ≥ 0, X₂ ≥ 0, X₃ = 1 − X₁ − X₂ ≥ 0 cut out a triangle of admissible portfolios in that plane. Now what do E and V look like as functions of (X₁, X₂)?

E = Σ Xᵢ μᵢ is linear in the weights, so after substituting X₃ it is an affine function of (X₁, X₂): E = μ₃ + X₁(μ₁ − μ₃) + X₂(μ₂ − μ₃). The set of portfolios with a fixed expected return E₀ is therefore a straight line — set that affine expression equal to E₀ and you get a line in the (X₁, X₂) plane. Change E₀ and the line moves but keeps its slope, because changing the constant only changes the intercept. So the iso-expected-return lines — call them isomean lines — are a family of parallel straight lines, marching across the triangle in the direction of increasing E.

V = Σ Σ Xᵢ Xⱼ σᵢⱼ, after the same substitution, is a quadratic in (X₁, X₂), and in the non-degenerate case it is a positive quadratic — variance is never negative, and two distinct portfolios do not have perfectly locked returns — so its level sets, the iso-variance curves, are concentric ellipses. They are centered on the single point that minimizes V over the whole plane; call it X̄, the global minimum-variance portfolio. As you move away from X̄ in any direction, V grows, and the ellipse you're on gets bigger. So the picture in the (X₁, X₂) plane is: parallel straight isomean lines in one direction, nested ellipses of variance around X̄.

Finding the efficient set is now a geometry problem on this picture. Fix an expected return E₀; its portfolios are one isomean line. Among the portfolios on that line, the efficient one is the one with the smallest variance — the point where the line touches the smallest ellipse it can, i.e. where the isomean line is *tangent* to an iso-variance ellipse. (If it crossed an ellipse, I could slide along the line to a smaller ellipse; tangency is where I can't.) Now let E₀ vary and watch that tangency point move. Each isomean line in the parallel family has its own tangency point with the ellipses, and as the lines sweep across (all parallel, so the tangency condition is the same linear condition each time), the tangency points trace out a *straight line* through X̄ — the locus of minimum-variance portfolios for each level of E. Call it the critical line.

So the minimum-variance-for-each-E portfolios lie on this critical line, but the triangle still matters. If X̄ lies inside the triangle, the efficient set begins there; if X̄ lies outside, it begins at the attainable boundary point with the smallest variance. From that minimum-variance point I move in the direction of increasing E. Along a critical-line segment, every step buys more E at the cost of more V, so those points are efficient; moving the other way costs more V for less E, so those points are dominated. When the line hits the edge of the triangle, a no-short-sale constraint binds, some Xᵢ hits zero, and the efficient set continues along the boundary or along the critical line of that lower-dimensional face, turning again when another constraint binds. It stops at a maximum-E attainable portfolio, usually the all-in-the-best-mean vertex I started by rejecting. That concentrated portfolio is one aggressive endpoint of the frontier, not the whole answer. The efficient set is the connected chain of straight segments from minimum available variance to maximum attainable expected return.

And the same structure holds with more securities, just in higher dimension. With four securities, Σ Xᵢ = 1 drops me into three-space, the admissible set is a tetrahedron, and for each subset of securities (each face/subspace where some Xᵢ = 0) there is a critical line; the efficient set is traced by starting at the minimum available variance point and moving along these critical lines according to which constraints are active, turning whenever a line meets a boundary or a lower-dimensional subspace, and ending at the maximum-E point. A connected series of line segments again. So the procedure generalizes: it is a systematic walk from the safe end to the aggressive end, and it works for any number of securities.

Let me also see what the frontier looks like in the space the investor actually cares about, the (E, V) plane, because that is the curve I hand them. E is a linear function of the weights and V is a quadratic one. Over the efficient portfolios, as I move along a critical-line segment, the weights move linearly in a parameter, so E moves linearly and V — being quadratic in the weights — moves as a quadratic in that same parameter. Eliminating the parameter, V is a quadratic function of E along each segment: a parabola. With the no-short-sale simplex, plotting V against E for efficient portfolios gives connected parabola segments; plotting standard deviation σ = √V against E gives the familiar bullet-shaped frontier, with the minimum-variance portfolio at the leftmost point and a maximum-return attainable portfolio at the top.

Let me make this concrete with a small computation, so I can see the trade-off and confirm the diversification really shows up in numbers. Take three securities with expected returns μ = (0.06, 0.10, 0.14), standard deviations (0.10, 0.18, 0.28), and correlations ρ₁₂ = 0.30, ρ₁₃ = 0.20, ρ₂₃ = 0.50 — so the covariance matrix is Σᵢⱼ = ρᵢⱼ σᵢ σⱼ. For the clean algebra I temporarily keep only the two equality constraints, wᵀμ = E and wᵀ1 = 1; when a no-short-sale inequality binds, the same calculation is repeated on the active face. The Lagrangian is wᵀΣw − 2λ(wᵀ1 − 1) − 2γ(wᵀμ − E); setting the gradient to zero gives Σw = λ1 + γμ, so w = Σ⁻¹(λ1 + γμ). Imposing the two constraints gives a 2×2 linear system in (λ, γ) with the scalars A = 1ᵀΣ⁻¹1, B = 1ᵀΣ⁻¹μ, C = μᵀΣ⁻¹μ, and D = AC − B². Solving, λ = (C − BE)/D and γ = (AE − B)/D, and back-substituting gives the minimum-variance weights for each target E. The achieved variance is not another mystery: V = wᵀΣw = wᵀ(λ1 + γμ) = λ + γE = (A E² − 2BE + C)/D. If I minimize variance without fixing E, the same first-order condition with only wᵀ1 = 1 gives the global minimum-variance portfolio w_mv = Σ⁻¹1 / (1ᵀΣ⁻¹1). Sweeping E traces the equality-constrained parabola; with no short sales, the moments where a weight would go negative are exactly where the connected-segment frontier turns onto a boundary.

```python
import numpy as np

def portfolio_mean(weights, mu):
    # expected return of a weighted sum = weighted sum of expected returns
    return weights @ mu

def portfolio_variance(weights, Sigma):
    # variance of a weighted sum carries all the covariance cross terms:
    # V = sum_i sum_j w_i w_j Sigma_ij = w^T Sigma w
    return weights @ Sigma @ weights

def select_portfolios(mu, Sigma, targets):
    # for each target expected return, solve the minimum-variance equality problem
    one = np.ones(len(mu)); Si = np.linalg.inv(Sigma)
    A = one @ Si @ one          # these four scalars define the critical line
    B = one @ Si @ mu
    C = mu  @ Si @ mu
    D = A * C - B * B
    rows = []
    for E in targets:
        lam = (C - B * E) / D
        gam = (A * E - B) / D
        w = Si @ (lam * one + gam * mu)     # minimum-variance weights for E
        rows.append((E, portfolio_variance(w, Sigma), w))
    return rows, (A, B, C, D)

mu = np.array([0.06, 0.10, 0.14])
sd = np.array([0.10, 0.18, 0.28])
rho = np.array([[1.0, 0.30, 0.20],
                [0.30, 1.0, 0.50],
                [0.20, 0.50, 1.0]])
Sigma = np.outer(sd, sd) * rho

# global minimum-variance portfolio: w_mv = Si 1 / (1^T Si 1)
Si = np.linalg.inv(Sigma); one = np.ones(3)
w_mv = Si @ one / (one @ Si @ one)
# the rule I rejected -- everything in the max-mean security -- has sigma = 0.28;
# the min-variance mix has far less risk:
print("min-variance:  E=%.4f  sigma=%.4f  w=%s"
      % (portfolio_mean(w_mv, mu), np.sqrt(portfolio_variance(w_mv, Sigma)),
         np.round(w_mv, 3)))           # -> sigma ~ 0.0965, not 0.28

# covariance at work: an equal-weight mix has LOWER sigma than the weighted
# average of the stand-alone sigmas, because rho_ij < 1.
w_eq = one / 3
print("equal mix sigma=%.4f  vs weighted-avg stand-alone sigma=%.4f"
      % (np.sqrt(portfolio_variance(w_eq, Sigma)), w_eq @ sd))  # 0.1465 < 0.1867

# trace the equality-constrained branch up to the max standalone mean level
targets = np.linspace(portfolio_mean(w_mv, mu), 0.14, 7)
rows, (A, B, C, D) = select_portfolios(mu, Sigma, targets)
for E, V, w in rows:
    print("E=%.4f  sigma=%.4f  w=%s" % (E, np.sqrt(V), np.round(w, 3)))
# along the frontier V is a parabola in E:  V(E) = (A E^2 - 2 B E + C) / D
```

So the path, end to end: I read Williams and lifted "value = expected discounted dividends" into a portfolio rule, maximize Σ Xᵢ Rᵢ; that objective is a weighted average, maximized by putting everything in one security, which never diversifies — and since everyone diversifies, the rule is wrong, and it is wrong because it kept only the center of each security's distribution and deleted the spread, which is the very thing uncertainty makes investors care about. The law-of-large-numbers patch fails because returns are correlated, so dispersion has a floor and the max-return portfolio isn't the min-dispersion one — there's a real trade-off rate. Putting dispersion into the objective as variance, and writing the variance of the weighted sum, surfaces the covariance cross terms, and those terms are exactly what makes a mix of imperfectly correlated securities less variable than its parts — so variance is the right risk measure because diversification is built into it, and the right diversification is into low-covariance securities. With two opposed criteria and no assumed taste, the answer is the efficient set: minimum variance for each level of expected return. Geometrically, parallel isomean lines tangent to concentric iso-variance ellipses trace a critical line, and the efficient frontier runs along connected segments from the minimum-variance portfolio up to the maximum-return one — a single bullet-shaped curve in (σ, E) the investor chooses a point on.
