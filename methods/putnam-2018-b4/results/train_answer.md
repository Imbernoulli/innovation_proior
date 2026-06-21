I will solve the problem using the Fibonacci–cosine substitution for a quadratic recurrence. The sequence is defined by $x_0 = 1$, $x_1 = x_2 = a$, and $x_{n+1} = 2 x_n x_{n-1} - x_{n-2}$ for $n \ge 2$, and I must prove that if some term $x_n$ equals zero, then the entire sequence is periodic.

My first step is to understand when a zero can even occur. If $|a| > 1$, I claim the sequence never reaches zero. I prove by induction that $|x_{n+1}| \ge |x_n| \ge 1$ for every $n$. The base cases are clear: $|x_0| = 1$ and $|x_1| = |x_2| = |a| > 1$, so $|x_2| \ge |x_1| \ge |x_0| = 1$. For the inductive step, assume $|x_k| \ge |x_{k-1}| \ge \cdots \ge |x_0| = 1$ for all $k \le n$. Then by the triangle inequality, $|x_{n+1}| = |2 x_n x_{n-1} - x_{n-2}| \ge 2 |x_n| |x_{n-1}| - |x_{n-2}|$. Because the magnitudes are nondecreasing, $|x_{n-2}| \le |x_n|$, so this is at least $|x_n| (2 |x_{n-1}| - 1) \ge |x_n|$. Thus the magnitudes stay at least $1$ and never decrease, so no term can be zero when $|a| > 1$. Consequently, any zero forces $|a| \le 1$.

Once $|a| \le 1$, I can write $a = \cos b$ for some real angle $b$ in $[0, \pi]$. The recurrence now invites a trigonometric interpretation. Computing the first few terms as polynomials in $a$ gives $x_3 = 2a^2 - 1$, $x_4 = 4a^3 - 3a$, and $x_5 = 16a^5 - 20a^3 + 5a$. These are precisely the Chebyshev-like expressions for $\cos 2b$, $\cos 3b$, and $\cos 5b$. The multipliers $0, 1, 1, 2, 3, 5$ that appear for $x_0$ through $x_5$ are the Fibonacci numbers $F_0, F_1, F_2, F_3, F_4, F_5$, where $F_0 = 0$, $F_1 = F_2 = 1$, and $F_{n+1} = F_n + F_{n-1}$. I conjecture that $x_n = \cos(F_n b)$ for all $n$.

I verify this by induction. The base cases $n = 0, 1, 2$ follow from $\cos 0 = 1$ and $\cos b = a$. Assuming $x_k = \cos(F_k b)$ for all $k \le n$, I use the product-to-sum identity $2 \cos \alpha \cos \beta = \cos(\alpha + \beta) + \cos(\alpha - \beta)$. Substituting $\alpha = F_n b$ and $\beta = F_{n-1} b$ gives $2 x_n x_{n-1} = \cos((F_n + F_{n-1}) b) + \cos((F_n - F_{n-1}) b) = \cos(F_{n+1} b) + \cos(F_{n-2} b)$, where the last equality uses both the Fibonacci recurrence and its backward form $F_n - F_{n-1} = F_{n-2}$. Therefore $x_{n+1} = 2 x_n x_{n-1} - x_{n-2} = \cos(F_{n+1} b) + \cos(F_{n-2} b) - \cos(F_{n-2} b) = \cos(F_{n+1} b)$. The induction closes, and the recurrence is exactly the additive structure of Fibonacci indices transported through the cosine.

Now suppose $x_n = 0$ for some $n$. Since $x_0 = 1$, we have $n \ge 1$ and $F_n \ge 1$. From $x_n = \cos(F_n b) = 0$, the angle $F_n b$ must be an odd multiple of $\pi/2$, say $F_n b = k \pi/2$ for some odd integer $k$. Solving for $b$ gives $b = k \pi / (2 F_n)$. To make periodicity transparent, I write this as $b = (c/d) \cdot 2\pi$ where $c = k$ and $d = 4 F_n$. Then every term has the form $x_m = \cos(F_m b) = \cos((F_m c / d) \cdot 2\pi)$, so $x_m$ depends only on the residue of $F_m$ modulo $d$.

The final ingredient is that Fibonacci numbers are periodic modulo any fixed integer $d$. Consider consecutive pairs $(F_m, F_{m+1})$ as elements of $(\mathbb{Z}/d\mathbb{Z})^2$. This set has only $d^2$ elements, so some pair repeats: $(F_{n_1}, F_{n_1+1}) \equiv (F_{n_2}, F_{n_2+1}) \pmod d$ with $n_1 < n_2$. The forward recurrence $(u, v) \mapsto (v, u + v)$ is invertible on this finite set, with inverse $(v, w) \mapsto (w - v, v)$. Because the map is a bijection, the repeated pair propagates backward all the way to the start: with $\ell = n_2 - n_1$, we have $(F_0, F_1) \equiv (F_\ell, F_{\ell+1}) \pmod d$, and hence $(F_m, F_{m+1}) \equiv (F_{m+\ell}, F_{m+\ell+1}) \pmod d$ for all $m \ge 0$. In particular, $F_{m+\ell} \equiv F_m \pmod d$ for every $m$.

Returning to the sequence, for each $m$ the difference $F_{m+\ell} - F_m$ is divisible by $d$, so $(F_{m+\ell} c / d) - (F_m c / d)$ is an integer. Since cosine has period $2\pi$, $x_{m+\ell} = \cos((F_{m+\ell} c / d) \cdot 2\pi) = \cos((F_m c / d) \cdot 2\pi + \text{integer} \cdot 2\pi) = \cos((F_m c / d) \cdot 2\pi) = x_m$. Thus $x_{m+\ell} = x_m$ for all $m \ge 0$, which means the sequence is periodic with period $\ell$.

The same idea can be expressed for $|a| > 1$ by allowing complex angles: if $a > 1$, write $a = \cosh t$ and use $b = i t$, so every term is $\cosh(F_n t)$; if $a < -1$, write $a = -\cosh t$ and use $b = \pi + i t$, giving terms $(-1)^{F_n} \cosh(F_n t)$. In either case the magnitudes stay above $1$, consistent with the real induction and confirming that zeros only arise in the real-angle regime.

The Python script below verifies the entire argument on a concrete example. It defines the recurrence, compares it with the cosine formula for $a = \sqrt{2}/2$ (where $x_3 = 0$), and confirms that the resulting sequence repeats with the period predicted by the Pisano period of $d = 4 F_3 = 8$.

```python
import math

def recurrence_seq(a, n_terms):
    x = [0.0] * n_terms
    x[0] = 1.0
    if n_terms > 1:
        x[1] = a
    if n_terms > 2:
        x[2] = a
    for n in range(2, n_terms - 1):
        x[n + 1] = 2 * x[n] * x[n - 1] - x[n - 2]
    return x

def fibonacci(n_terms):
    F = [0] * n_terms
    if n_terms > 1:
        F[1] = 1
    if n_terms > 2:
        F[2] = 1
    for n in range(2, n_terms - 1):
        F[n + 1] = F[n] + F[n - 1]
    return F

# Choose a value where a zero occurs: a = cos(pi/4), so x_3 = 0.
a = math.sqrt(2) / 2
b = math.acos(a)  # pi/4
N = 30
F = fibonacci(N)
x_rec = recurrence_seq(a, N)
x_cos = [math.cos(F[n] * b) for n in range(N)]

print("Verify x_n = cos(F_n b) for n = 0..{}:".format(N - 1))
print(all(abs(x_rec[n] - x_cos[n]) < 1e-9 for n in range(N)))

# The zero at n=3 gives d = 4 * F_3 = 8; Pisano period modulo 8 is 12.
d = 4 * F[3]
period = 12
print("d =", d, "period =", period)
print("Check periodicity x[m+period] == x[m] for all m:",
      all(abs(x_rec[m + period] - x_rec[m]) < 1e-9 for m in range(N - period)))
```

This completes the proof: whenever one term of the sequence vanishes, the Fibonacci–cosine substitution forces the defining angle to be a rational multiple of $2\pi$, and the modular periodicity of Fibonacci numbers makes the sequence repeat.
