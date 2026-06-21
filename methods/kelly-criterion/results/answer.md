# Kelly Criterion

The method is to choose the fraction of current wealth that maximizes the long-run exponential growth rate of reinvested capital:

```text
G = lim_{N -> infinity} (1/N) log(V_N / V_0).
```

For a horse-race channel with outcome `s`, received signal `r`, gross payoff odds `alpha_s`, and fraction `a(s|r)` bet on outcome `s`, the growth rate is

```text
G = sum_{r,s} p(s,r) log(alpha_s a(s|r)).
```

With fair odds `alpha_s = 1/p(s)`, the optimal allocation is

```text
a*(s|r) = p(s|r),
```

and the maximum growth rate is Shannon's information rate:

```text
G_max = H(S) - H(S|R) = I(S;R).
```

For a binary wager with win probability `p`, loss probability `q = 1 - p`, and net odds `b:1`, staking fraction `f` gives

```text
g(f) = p log(1 + b f) + q log(1 - f).
```

The unique interior maximizer is

```text
f* = (b p - q) / b = p - q / b.
```

With no forced betting and no shorting:

```text
f* = max(0, (b p - q) / b).
```

For even-money odds, `b = 1`, so `f* = 2p - 1`.

This is not the one-period expected-gain rule. It is the allocation that matches capital fractions to probabilistic information so that the repeated wealth process grows at the maximum information-compatible rate.
