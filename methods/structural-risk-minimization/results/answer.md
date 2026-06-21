# Structural Risk Minimization

Structural Risk Minimization selects a predictor by minimizing a finite-sample upper bound on true risk rather than training loss alone.

Given a sample of size `l`, define expected and empirical risk:

`R(alpha) = integral L(y, f(x, alpha)) dP(x,y)`

`R_emp(alpha) = (1/l) sum_i L(y_i, f(x_i, alpha))`

Choose a nested structure of hypothesis spaces:

`S1 subset S2 subset ... subset Sn`

with increasing capacities, commonly VC dimensions:

`h1 < h2 < ... < hn`.

For each level, solve empirical risk minimization:

`alpha_k = argmin_{alpha in Sk} R_emp(alpha)`.

Then choose the level with the smallest guaranteed risk:

`k* = argmin_k [R_emp(alpha_k) + Omega(h_k, l, eta_k)]`.

Return `alpha_{k*}`.

For binary classification, Vapnik's 1991 bound gives the additive confidence term

`Omega_0(h,l,eta) = sqrt((h(ln(2l/h)+1) - ln eta) / l)`,

so with probability at least `1 - eta`,

`P(w) < v(w) + Omega_0(h,l,eta)`.

The method's two-step logic is:

1. Minimize empirical risk separately inside each capacity-controlled class.
2. Select the class whose empirical risk plus capacity/confidence penalty is smallest.

With countably many classes, assign weights `w(k)` with `sum_k w(k) <= 1` and use `eta_k = w(k) eta` so the bounds hold simultaneously across the structure. The selected predictor then satisfies an oracle-style guarantee: it competes with the best predictor in each candidate class plus that class's confidence penalty.

For margin classifiers, the structure can be induced by margin. If inputs lie in a ball of radius `R` and a hyperplane separates with margin `Delta`, Vapnik's overview gives

`h <= min(R^2 / Delta^2, n) + 1`.

Maximizing margin therefore reduces capacity. A hard-margin support vector machine is an implementation of the principle in the separable case: it keeps empirical error zero while choosing the separator from the smallest-capacity margin class. Soft-margin and kernel variants preserve the same tradeoff between empirical violations and capacity control.
