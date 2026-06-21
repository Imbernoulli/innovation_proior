I start with the only quantity I can compute. The distribution that generates future examples is hidden, so the true risk of a rule is an expectation I cannot evaluate. The sample gives me an average loss, and for a rule fixed in advance this average is a legitimate estimate. That fixed-rule statement is tempting enough to mislead me: if every fixed rule has its sample loss approach its true loss, why not choose the rule with the smallest sample loss?

The problem is that the chosen rule is not fixed in advance. I select it because the sample makes it look good. That means the rule I care about is biased toward unusually low sample loss. A concentration bound for one fixed rule is the wrong object. I need a statement that survives the act of searching.

So I change the question. I do not ask whether `R_emp(f)` is close to `R(f)` for one `f`. I ask whether the largest gap over the whole class is small:

```text
sup_{f in F} |R(f) - R_emp(f)|.
```

If that quantity is small, then the data-selected rule is covered automatically. Let `f_hat` be the rule with the smallest observed loss, and let `f_F` be the best rule in the class under true risk. I can insert and subtract empirical risks:

```text
R(f_hat) - R(f_F)
  = R(f_hat) - R_emp(f_hat)
    + R_emp(f_hat) - R_emp(f_F)
    + R_emp(f_F) - R(f_F).
```

The middle term is nonpositive because `f_hat` minimizes empirical risk. The two remaining terms are each bounded by the class-wide gap. Therefore the excess risk is at most twice the uniform gap. This is the first real insight: optimization on the sample is justified only through a uniform comparison between sample risk and true risk.

Now I hit the next wall. A finite class is easy: I can apply a fixed-rule concentration inequality to every rule and pay for the union with a factor equal to the number of rules. The cost becomes logarithmic in the number of rules once I solve the bound for the error radius. But learning classes are rarely finite. Linear separators, neural-network functions, and many parametric families contain infinitely many functions. Counting functions directly gives me infinity and no guarantee.

I need to count something else. On a sample of `n` points, an infinite binary class cannot show more than `2^n` different label patterns. Many infinite classes show far fewer. The relevant size is not how many formulas or parameters the class has, but how many different dichotomies it can impose on finite samples. This turns the infinite search problem back into a finite combinatorial problem.

The ghost-sample move explains why this finite-sample count is legitimate. The true risk still depends on the unknown distribution, so I cannot directly replace the supremum over `R(f) - R_emp(f)` by a count of labelings on the observed sample. I introduce an independent second sample from the same distribution. If the first sample risk is far from true risk for some function, then with substantial probability the first and second empirical risks are far from each other. Now both quantities are computed on a finite set of `2n` sample points, and functions that agree on those points are indistinguishable for this comparison.

At that point a union bound can return, but it is a union bound over label patterns rather than over raw functions. The growth function or shattering coefficient becomes the effective class size. If this effective size grows only polynomially with `n`, the exponential concentration term wins and the uniform gap shrinks. If the class can realize all `2^n` labelings for arbitrarily large `n`, then it can keep matching arbitrary samples, and the sample loss no longer disciplines true risk.

This is where capacity enters. The largest sample size that a class can shatter summarizes whether its growth is polynomial or full exponential. Finite capacity means that after some sample size the class cannot realize every labeling; the number of observable patterns grows slowly enough for uniform convergence. Infinite capacity means no such obstruction exists, so a low sample loss may simply be memorization.

The method becomes clear only after these corrections. The operational rule is simple: choose the function in the class with the smallest empirical risk. The guarantee is not simple training-error optimism. The guarantee comes from a class-level uniform law: with high probability, every function's sample risk and true risk are close. Under that event, the empirical minimizer's true risk is close to the best true risk in the class.

I also need to be precise about what this does not solve. If the class is too small, the best rule inside it may still have high true risk. The empirical-risk rule only competes with the best member of the chosen class. If the class is too large, the uniform gap can be too large, and zero training error says little. The scientific content is the coupling of a computable objective with a capacity-controlled class.

So the final picture is not "fit the training set." It is: define true risk, replace it by empirical risk because that is observable, minimize the empirical risk within a fixed class, and require a uniform convergence theorem for that class so the data-dependent minimizer inherits a risk bound. The breakthrough is the shift from pointwise estimation to whole-class control. Once the class has finite effective capacity, empirical fitting becomes a defensible induction principle rather than a memorization recipe.
