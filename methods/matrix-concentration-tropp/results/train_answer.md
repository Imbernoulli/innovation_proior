I present the method of matrix concentration inequalities via Lieb cumulants, often associated with Tropp's unified treatment of matrix Chernoff, Bernstein, Bennett, and Hoeffding bounds. The problem is to control the largest eigenvalue of a sum of independent random self-adjoint matrices, or equivalently the spectral norm of a sum of independent rectangular matrices after self-adjoint dilation. Scalar concentration tools fail here because matrix exponentials do not multiply when summands fail to commute, and entrywise bounds do not capture spectral behavior. The key insight is to lift not the raw moment generating function but the cumulant generating function, using Lieb's concavity theorem as the noncommutative replacement for additivity of logarithms.

Begin with independent random self-adjoint matrices X_1, ..., X_n and form Y = sum_k X_k. The Laplace transform bound for the maximal eigenvalue is identical in spirit to the scalar case: for any theta > 0,

P{lambda_max(Y) >= t} = P{exp(theta lambda_max(Y)) >= exp(theta t)} <= exp(-theta t) E tr exp(theta Y).

The last inequality uses Markov's inequality and the fact that exp(theta lambda_max(Y)) = lambda_max(exp(theta Y)) <= tr exp(theta Y). At this point the scalar proof would split E exp(theta Y) into a product of one-dimensional mgfs. For matrices, exp(theta X_1 + ... + theta X_n) is not the product of the individual exponentials, so that route is blocked.

The escape is Lieb's theorem. For fixed self-adjoint H, the map A -> tr exp(H + log A) is concave on the positive definite cone. Taking A = exp(X) and applying Jensen's inequality gives

E tr exp(H + X) <= tr exp(H + log E exp(X)).

This is the crucial one-summand replacement. It does not multiply exponentials; instead it absorbs a single random matrix by passing to its matrix cumulant log E exp(X) while keeping everything else under one trace exponential. Iterate the step over the independent summands, conditioning on earlier variables and treating the remaining expression as H. Independence ensures that each conditional mgf equals the unconditional mgf. After n steps,

E tr exp(theta Y) <= tr exp(sum_k log E exp(theta X_k)).

This is matrix cgf subadditivity. It is weaker than scalar additivity but strong in the right way: the cumulant matrices are summed before any spectral norm or largest eigenvalue is taken. Combining with the Laplace bound gives the master inequality

P{lambda_max(Y) >= t} <= inf_{theta > 0} exp(-theta t) tr exp(sum_k log E exp(theta X_k)).

The rest is a calculus of semidefinite mgf bounds. Suppose for each summand we can show E exp(theta X_k) <= exp(g(theta) A_k) for some function g and fixed positive semidefinite A_k. Operator monotonicity of the matrix logarithm gives log E exp(theta X_k) <= g(theta) A_k. Then

tr exp(sum_k log E exp(theta X_k)) <= tr exp(g(theta) sum_k A_k) <= d exp(g(theta) lambda_max(sum_k A_k)),

where d is the ambient dimension. The variance scale is therefore lambda_max(sum_k A_k), the spectral norm of the summed variance matrices, rather than sum_k lambda_max(A_k). This distinction is the whole point: for heterogeneous sums the two can differ by the dimension, and keeping the sum outside the eigenvalue is what removes that loss from the exponent.

For a Gaussian or Rademacher matrix series sum_k xi_k A_k, the scalar subgaussian estimate lifts to matrices as E exp(theta xi A) <= exp(theta^2 A^2 / 2). The master bound then yields

P{lambda_max(sum_k xi_k A_k) >= t} <= d exp(-t^2 / (2 ||sum_k A_k^2||)).

A union over Y and -Y gives the two-sided operator norm bound. For rectangular sums sum_k xi_k B_k, one applies the self-adjoint dilation to convert singular values into eigenvalues; the variance parameter becomes the maximum of the row and column sums of squares, max{||sum_k B_k B_k^*||, ||sum_k B_k^* B_k||}, and the dimension factor becomes d_1 + d_2.

For bounded centered summands with E X_k = 0 and lambda_max(X_k) <= R, the exponential remainder is controlled by the monotonicity of (exp(theta x) - theta x - 1) / x^2 on [-R, R]. This produces the matrix Bernstein bound

P{lambda_max(sum_k X_k) >= t} <= d exp(-(t^2 / 2) / (sigma^2 + R t / 3)),

where sigma^2 = ||sum_k E X_k^2||. For positive semidefinite summands bounded in [0, R], a chord bound on exp(theta x) gives matrix Chernoff upper and lower tails in terms of the eigenvalues of sum_k E X_k. In every case the same Lieb-cumulant machinery supplies the noncommutative skeleton, and only the one-dimensional mgf estimate changes.

The method's canonical name is matrix concentration inequalities via Lieb cumulants, or Tropp-style matrix concentration. It is the natural noncommutative analogue of the scalar Laplace-transform method: the trace exponential provides the scalar functional, Lieb's concavity theorem provides additivity of cumulants, and the variance matrices are summed before any spectral norm is applied.

The final form of the result is the master tail bound together with its most useful corollaries. For independent self-adjoint matrices $X_1,\dots,X_n$ with each conditional mgf bounded by $\mathbb{E}\exp(\theta X_k) \preceq \exp(g(\theta) A_k)$, the master inequality is

$$\mathbb{P}\{\lambda_{\max}(Y) \ge t\} \le \inf_{\theta>0} \exp(-\theta t)\,\operatorname{tr}\exp\Big(\sum_k \log \mathbb{E}\exp(\theta X_k)\Big) \le d \inf_{\theta>0} \exp\big(-\theta t + g(\theta)\,\lambda_{\max}(\textstyle\sum_k A_k)\big).$$

For a Gaussian or Rademacher series $\sum_k \xi_k A_k$ with fixed self-adjoint $A_k$, the subgaussian estimate $g(\theta) = \theta^2/2$, $A_k \to A_k^2$ turns this into

$$\mathbb{P}\{\lambda_{\max}(\textstyle\sum_k \xi_k A_k) \ge t\} \le d\exp\!\left(-\frac{t^2}{2\,\|\sum_k A_k^2\|}\right),$$

and for a rectangular series $\sum_k \xi_k B_k$ the self-adjoint dilation converts this into

$$\mathbb{P}\{\|\textstyle\sum_k \xi_k B_k\| \ge t\} \le (d_1+d_2)\exp\!\left(-\frac{t^2}{2\sigma^2}\right), \qquad \sigma^2 = \max\Big\{\big\|\textstyle\sum_k B_k B_k^*\big\|,\ \big\|\textstyle\sum_k B_k^* B_k\big\|\Big\}.$$

For centered summands with $\mathbb{E}X_k = 0$ and $\lambda_{\max}(X_k) \le R$ almost surely, the exponential-remainder estimate on $[-R,R]$ gives the matrix Bernstein bound

$$\mathbb{P}\{\lambda_{\max}(\textstyle\sum_k X_k) \ge t\} \le d\exp\!\left(-\frac{t^2/2}{\sigma^2 + Rt/3}\right), \qquad \sigma^2 = \Big\|\sum_k \mathbb{E}X_k^2\Big\|,$$

and for positive semidefinite summands bounded in $[0,R]$, the same calculus with a chord bound on $\exp(\theta x)$ produces the matrix Chernoff upper and lower tails in terms of the eigenvalues of $\sum_k \mathbb{E}X_k$. In every one of these corollaries the dimension $d$ (or $d_1+d_2$) is the only trace of ambient dimension, and it enters as a polynomial prefactor rather than inside the exponent — the entire content of the Lieb-cumulant argument is that the exponent depends on the matrices only through $\lambda_{\max}$ of their summed variance, never through a sum of per-summand eigenvalues.
