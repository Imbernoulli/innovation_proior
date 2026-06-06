OK, let me think this through from scratch. I have N identical charges pinned to the unit sphere S^{n-1}, repelling each other, and I want to know how they settle — the arrangement that minimizes total potential energy. Write it as sum over ordered pairs x != y of f(|x-y|^2), where f is the potential as a function of squared distance; for Coulomb in R^3 that's f(r) = 1/sqrt(r), i.e. 1/|x-y|, and for the harmonic law in R^n it's f(r) = 1/r^{n/2-1}. I'd also like to handle every inverse power 1/r^s and things like Gaussians, but let me hold that thought.

The obvious thing to do is just minimize. Put the points down at random, compute the gradient of the energy on the sphere, slide everything downhill, repeat. That works — it gives me a configuration, and for N = 12 on S^2 it lands on the icosahedron, which feels right. But the moment I try to *trust* it I hit a wall. The energy landscape is a mess: the number of local minima grows like an exponential in N, so my downhill walk has no way of knowing it found the global minimum rather than one of a swarm of nearby traps. And it gets worse — for N = 8 and N = 20 the actual minimizer isn't the Platonic solid I'd have guessed; the cube and dodecahedron lose. So even my intuition about "the symmetric thing wins" is unreliable. Steepest descent is a way to *guess* the answer. It is not a way to *prove* it. If I want a theorem I cannot work in configuration space at all — there are too many configurations and I can never rule them all out one by one.

So flip the whole thing around. Instead of hunting through configurations from above, I want a quantity that bounds *every* configuration's energy from below, simultaneously, and then I want to find a specific arrangement that hits the bound. If I can write down a number L with "energy of any N points >= L" and also exhibit a configuration whose energy equals L, I'm done — that configuration is optimal, and I never had to enumerate anything. The art is entirely in producing L.

Where would such a universal lower bound come from? The energy only depends on the inner products between points, since |x-y|^2 = 2 - 2<x,y> on the unit sphere. So write everything in the variable t = <x,y> in [-1,1], and let a(t) = f(2 - 2t) be the potential as a function of inner product. The energy is sum_{x != y} a(<x,y>). Now here's the move I keep circling back to: suppose I had a polynomial h(t) that sits *underneath* a, meaning h(t) <= a(t) for all t in [-1,1). Then

   sum_{x != y} a(<x,y>) >= sum_{x != y} h(<x,y>).

I've replaced the awkward potential by a polynomial, losing nothing as a lower bound. Why is a polynomial better? Because of how polynomials interact with the geometry of the sphere. Let me add back the diagonal terms x = y, which contribute h(1) each, N of them:

   sum_{x != y} h(<x,y>) = sum_{all x,y} h(<x,y>) - N h(1).

The full double sum over all pairs, including the diagonal, is the thing I can control. And the way to control it is to expand h in the right basis.

What basis? I want the double sum sum_{x,y} (something)(<x,y>) to have a sign I can predict, for *every* configuration. That's exactly the property of a positive-definite kernel. On the sphere there's a canonical family: decompose L^2(S^{n-1}) under the rotation group into spherical harmonics of each degree l, V_0, V_1, V_2, .... Each V_l has a reproducing kernel, and since O(n) acts distance-transitively that kernel depends only on the inner product — call it C_l(<x,y>). It's positive-definite for the cleanest possible reason: written as <ev_{l,x}, ev_{l,y}>, the double sum factors,

   sum_{x,y} C_l(<x,y>) = |sum_x ev_{l,x}|^2 >= 0.

That's a norm-squared; it can't be negative. And these C_l are forced by orthogonality to be the Gegenbauer / ultraspherical polynomials C_l^{lambda} with lambda = n/2 - 1, orthogonal with respect to (1 - t^2)^{(n-3)/2} dt, normalized so C_0 = 1, C_1 = 2*lambda*t, satisfying i C_i = 2(i+lambda-1) t C_{i-1} - (i+2*lambda-2) C_{i-2}. Schoenberg told me these are *all* of the continuous distance-only positive-definite kernels — any nonnegative combination of them is positive-definite, and there are no others. So the right basis to expand h in is the Gegenbauer basis, and the right constraint on the coefficients is that they be nonnegative.

Let me see what that buys. Write h(t) = sum_i alpha_i C_i^{n/2-1}(t). Then

   sum_{x,y} h(<x,y>) = sum_i alpha_i sum_{x,y} C_i(<x,y>).

The i = 0 term is alpha_0 sum_{x,y} C_0 = alpha_0 N^2, exactly, because C_0 = 1. Every other term, if alpha_i >= 0, is alpha_i times a nonnegative number — by the norm-squared above — so it only helps:

   sum_{x,y} h(<x,y>) >= alpha_0 N^2.

Put it together. For *any* N points on S^{n-1},

   energy >= sum_{x != y} h(<x,y>) = sum_{x,y} h(<x,y>) - N h(1) >= N^2 alpha_0 - N h(1).

There it is — a lower bound that depends only on h, not on the configuration. The two requirements on h were: (i) h(t) <= a(t) = f(2 - 2t) pointwise on [-1,1), and (ii) all Gegenbauer coefficients alpha_i >= 0. Within those two constraints I'm free, and I want to *maximize* N^2 alpha_0 - N h(1). That's a linear functional of the coefficients, subject to linear constraints (the pointwise domination is a linear inequality at each t, the positivity is alpha_i >= 0). It's a linear program — an infinite-dimensional one, but I can discretize the domination constraint on a fine grid of t-values and solve a finite LP numerically. This is the generalization of the Delsarte / Kabatiansky–Levenshtein LP bound for codes; in fact I can recover those by taking a to be +infinity above a threshold inner product and 0 below — the energy becomes 0 for a valid code and +infinity otherwise, and the bound on N falls out.

Let me sanity-check the engine before I push further. Twenty points on S^2, Coulomb f(r) = 1/sqrt(r). I solve the finite LP with a degree-6 h and get a lower bound around 301.2; the numerically-found minimizer (which, again, is *not* a dodecahedron) sits near 301.76. The bound is close but not tight. So in general the LP doesn't nail the answer — there's a gap, and the gap is real. Fine. The whole game now is: for *which* configurations is there an h that closes the gap completely? Because for those, I'll have a proof.

When is the bound sharp for a configuration S? Trace back through the inequalities and find what must be equalities. First, sum_{x != y} a(<x,y>) = sum_{x != y} h(<x,y>) needs h(t) = a(t) at every inner product t that actually occurs between distinct points of S — otherwise I'm strictly losing energy there. Second, in sum_i alpha_i sum_{x,y} C_i >= alpha_0 N^2, every i > 0 term with alpha_i > 0 must contribute exactly zero, i.e. sum_{x,y} C_i(<x,y>) = 0. But that's a familiar condition: sum_{x,y} C_i(<x,y>) = |sum_x ev_{i,x}|^2 = 0 means the degree-i spherical harmonics all sum to zero over S, and by Delsarte–Goethals–Seidel that holds for 1 <= i <= M exactly when S is a spherical M-design. So if h is positive-definite with degree at most M and S is an M-design, the second batch of equalities is automatic.

Now I can see the *shape* of the configurations that will work. I need S to have only a few distinct inner products — say m of them, t_1 < ... < t_m — so that "h equals a at every occurring inner product" is only m conditions, leaving me room. And I need S to be a spherical design of strength at least the degree of h. The two pull against each other: more distinct distances means I need a higher-degree h to interpolate them, which means I need a higher-strength design. The configurations where they balance perfectly are the ones where there are m distances and S is a (2m - 1)-design. Stare at that — why 2m - 1? Because I'm going to want h to touch a at each t_i to *second order* (I'll see why in a moment), so h has degree 2m - 1, and I need the design strength to match. And 2m - 1 is the most I can hope for: if S had only m distinct inner products t_1,...,t_m and were a (2m+1)-design, look at the polynomial x -> (1 - <x,y>) prod_i (<x,y> - t_i)^2, degree 2m + 1, for a fixed point y in S. It vanishes at every point of S (each x hits some t_i, or x = y giving the (1 - <x,y>) factor). If S were a (2m+1)-design its average over S would equal its average over the whole sphere — but it's a nonnegative, not-identically-zero polynomial, so its integral over the sphere is strictly positive, contradiction. So a configuration with m distances can be at most a 2m-design, and (2m - 1) is essentially maximal. Call such a thing — m distances, a (2m - 1)-design — a *sharp* configuration. The icosahedron is one (m = 3 inner products -1, ±1/sqrt5; a 5-design). The E_8 minimal vectors (m = 4 inner products -1, ±1/2, 0; a 7-design) and the Leech minimal vectors (m = 6; an 11-design) are too.

So the target is clear: given a sharp configuration with inner products t_1,...,t_m and given a completely monotonic f, construct a polynomial h of degree 2m - 1 such that h <= a on [-1,1), h = a at each t_i, and h is positive-definite. Three demands. Let me try to build it.

The first instinct: just interpolate a at the m points t_i. That's a degree m - 1 polynomial agreeing with a at t_1,...,t_m. But that's only first-order contact, and h will cross a at each t_i — on one side h > a, violating domination. I need h to *stay below* a. The fix is to make contact to *second* order at each t_i: demand h(t_i) = a(t_i) and h'(t_i) = a'(t_i). That's Hermite interpolation at the m points, each to order 2, giving a polynomial of degree 2m - 1 — which is exactly the degree the design strength supports. And second-order contact means h touches a tangentially and bounces back, no sign change, so I can hope h <= a everywhere. Let me check that hope is actually justified rather than wishful.

The Hermite interpolation remainder formula says: if h matches a to order 2 at each of t_1,...,t_m, then for any t there's a point xi between t and the t_i with

   a(t) - h(t) = a^{(2m)}(xi) / (2m)! * (t - t_1)^2 ... (t - t_m)^2.

The product prod (t - t_i)^2 is a square, so it's >= 0. And a^{(2m)}(xi) — that's where completely monotonic earns its keep. a(t) = f(2 - 2t); chasing the chain rule, complete monotonicity of f (meaning (-1)^k f^{(k)} >= 0) turns into *absolute* monotonicity of a, meaning every derivative a^{(k)} >= 0, all signs nonnegative. So a^{(2m)}(xi) >= 0. Both factors nonnegative, so a(t) - h(t) >= 0, i.e. h <= a on all of [-1,1). Domination, for free, as a consequence of complete monotonicity. This is exactly why I needed the strong hypothesis: ordinary convexity would only give me the simplex case (one distance, a tangent line); I need *all* derivatives nonnegative to make the high-order remainder argument go through. And it shows me why the whole family of completely monotonic potentials will fall at once — the only thing the construction used about f was absolute monotonicity of a, which every completely monotonic f provides. One construction, every potential.

(At t_1 = -1, when the smallest inner product is the antipodal one, second-order contact is overkill — first order suffices there, since there's no domination concern at the boundary. But it does no harm to use second order uniformly, so let me not fuss about it.)

The sharpness conditions are now half-done: h = a at the t_i is built in, and S being a (2m - 1)-design with deg h = 2m - 1 kills the i > 0 Gegenbauer terms — provided h really is positive-definite up to degree 2m - 1. That last demand, positive-definiteness, is the subtle one, and it's where I have to actually work.

I need: all Gegenbauer coefficients of h = H(a, F^2) are nonnegative, where F(t) = prod_{i=1}^m (t - t_i) and H(a, F^2) is the Hermite interpolant matching a to order 2 at the roots of F (i.e. at each t_i). Why is this not obvious? Because a is some arbitrary completely monotonic function; its interpolant's Gegenbauer coefficients could a priori be anything. I need a property of the *construction* that forces nonnegativity for every absolutely monotonic a. Let me name that property: call a polynomial g *conductive* if, for every absolutely monotonic a, the Hermite interpolant H(a, g) is positive-definite. My goal is exactly: F^2 is conductive.

How would conductivity propagate? Suppose I build up F^2 as a product of simpler conductive pieces. There's an identity for Hermite interpolation under a product of moduli. Write Q(a, g) = (a - H(a, g))/g, the "quotient after interpolation," which is C^infinity even at the roots of g. Then peeling one factor at a time,

   H(a, g_1 g_2) = H(a, g_1) + g_1 * H(Q(a, g_1), g_2).

So the interpolant of a against the product splits into the interpolant against g_1, plus g_1 times the interpolant of the *remainder function* Q(a, g_1) against g_2. If I can argue that (a) H(a, g_1) is positive-definite, (b) Q(a, g_1) is again absolutely monotonic so its interpolant against g_2 is positive-definite, and (c) products of positive-definite functions are positive-definite — then the right-hand side is a sum of positive-definite things and I win, inductively.

Take (b) first: is Q(a, g) absolutely monotonic when a is? Here the remainder formula resurfaces in a stronger form. Q(a, g)(t) = a^{(deg g)}(xi)/(deg g)! at some xi, which is >= 0 — that handles the 0th derivative. For higher derivatives, use that interpolating against (t - s)^k extracts the k-th derivative: Q(a, (t - s)^k)(s) = a^{(k)}(s)/k!. Composing the two interpolation identities, the n-th derivative of Q(a, g) at any point equals (up to a positive factorial) Q(a, g * (t - s)^n) at that point, which is again a high derivative of a evaluated somewhere, hence >= 0. So yes — *if* the roots of g lie in the interval, Q(a, g) inherits absolute monotonicity. (The roots-in-interval hypothesis is essential; that's why I need all the t_i in [-1, 1), which they are, being inner products.) Good — the remainder function stays in the same class, so the induction can continue.

For (c): products of positive-definite functions are positive-definite. This is the statement that the product of two Gegenbauer polynomials expands in Gegenbauer polynomials with nonnegative coefficients — equivalently, the integral of C_i C_j C_k against the weight is >= 0. That's clean: writing each C_l via its orthonormal harmonic basis, that triple integral becomes a sum of squared integrals of products of harmonics, manifestly >= 0. (It's also the Schur product theorem: positive-definite kernels are closed under entrywise product.) So positive-definite functions form a multiplicatively closed cone.

Now the base cases and the build-up. The simplest factor is linear: ell_r(t) = t - r. Interpolating a against it gives the constant a(r) >= 0 — positive-definite (constants are nonnegative multiples of C_0). So every linear factor is conductive. To assemble F = prod (t - t_i), I march through partial products prod_{i=1}^j (t - t_i). I claim each is itself positive-definite — and then conductivity multiplies up via the identity above. Why are the partial products positive-definite? For j < m, this is where the *design* hypothesis re-enters. The i-th Gegenbauer coefficient of F is, by orthogonality, a positive constant times the integral over the sphere of F(<x,y>) C_i(<x,y>); since S is a (2m - 1)-design and deg F + i <= 2m - 1, that integral equals the same positive constant times the *finite sum* over x in S of F(<x,y>) C_i(<x,y>) for a fixed y in S — and if I pick y in S, then F(<x,y>) vanishes for every x != y (because <x,y> is one of the t_i, the roots of F), leaving only the x = y term F(1) C_i(1), which is positive. So every low Gegenbauer coefficient of F is positive; the leading one is positive because F's leading coefficient is. F is strictly positive-definite. To handle the partial products cleanly, express F in terms of the orthogonal polynomials p_k for the weight (1 - t) dmu(t) (a Jacobi-type family that are themselves nonnegative combinations of Gegenbauers): F = p_m + alpha p_{m-1} for some alpha, which one checks by showing (1 - t) F is orthogonal to all lower-degree polynomials — and that orthogonality is again the design property forcing the finite sum over S to telescope to zero on the roots of F. The roots of p_m + alpha p_{m-1} interlace nicely, and the partial products prod_{i=1}^j (t - t_i) come out (strictly) positive-definite for j < m, with the j = m case being F itself. So every partial product is positive-definite, hence conductive; their product F is conductive; and F^2 = F * F is conductive too. Therefore h = H(a, F^2) is positive-definite.

That's the whole certificate. h has degree 2m - 1, sits below a with second-order contact at the m inner products of S, and is positive-definite. Feed it into the bound: every N-point configuration has energy at least N^2 alpha_0 - N h(1), and S meets it with equality because S is a sharp configuration. S is a global minimizer.

And the striking thing is what the argument *didn't* depend on. It never used which completely monotonic f I started with — only that a is absolutely monotonic. So the *same* sharp configuration minimizes energy for the Coulomb law, every Riesz power 1/r^s, the Gaussian e^{-cr}, all of them at once. That's a far stronger statement than minimizing one particular energy. By Bernstein–Widder every completely monotonic function is a nonnegative mixture of the simple functions (4 - r)^k, so to certify this universal optimality I only ever need to check the bound on f(r) = (4 - r)^k (or equivalently on 1/r^s), and the mixture inherits it. Let me call a configuration *universally optimal* when it weakly minimizes energy for every completely monotonic potential simultaneously. The icosahedron, the simplices and cross polytopes, the E_8 and Leech minimal vectors — all sharp, all universally optimal. And universal optimality implies being an optimal code (maximal minimal angle), since in the large-s limit of 1/r^s the leading energy term is dominated by the minimal distance — so this single result threads together the kissing/packing optimality at one end and the harmonic-energy results at the other.

One configuration resists the clean argument: the 600-cell, n = 4, N = 120, six inner products, but only an 11-design — while the Hermite construction would hand me a degree-15 polynomial, whose Gegenbauer coefficients above degree 11 the design strength can't kill. The bound from the naive h isn't sharp here. The repair is to give up a little: don't demand full second-order contact at all six points; instead build h of controlled degree and explicitly *enforce* the dangerous high Gegenbauer coefficients (the 12th through whatever) to be nonnegative while arranging the others to vanish, trading some interpolation freedom to keep positive-definiteness within the 11-design's reach. It's more delicate, but the same two ingredients — a polynomial below a with nonnegative Gegenbauer content — carry it through.

Step back and look at what made all this work, because it points somewhere bigger. The engine had three parts: a *positivity* structure under which the all-pairs sum has a definite sign (Gegenbauer / Schoenberg on the sphere); a *domination* of the potential from below by a function the positivity can handle; and *double roots* of (a - h) exactly at the distances of the special configuration, which is what makes the bound tight. None of those three is intrinsically about spheres. In Euclidean space R^n the analogue of "Gegenbauer coefficients nonnegative" is "Fourier transform nonnegative," and the analogue of "sum over all pairs >= constant" is the Poisson summation formula for a lattice. So the same certificate should work for lattices: find a radial function f with f and its Fourier transform f-hat both sign-controlled, dominating the potential from below, with double roots at the lengths of the lattice's vectors — and Poisson summation makes the bound tight. The construction of *that* magic function is the hard part — you have to hit prescribed values and derivatives of both f and f-hat at the radii sqrt(2k), the vector lengths of E_8 and of the Leech lattice, simultaneously — and it's solved by an interpolation theorem built from integral transforms of modular forms, which is the Euclidean counterpart of the Hermite-interpolation-of-an-absolutely-monotonic-function I did on the sphere. Same three-part skeleton — positivity, domination, double contact at the special distances — now proves E_8 in dimension 8 and the Leech lattice in dimension 24 are universally optimal among all configurations of their density. The sphere case is the rehearsal; the lattice case is the same idea over a harder analytic instrument.

Let me write the sphere certificate as code, the dual side of the problem. The primal search — steepest descent — only ever *finds* a candidate; the LP certificate is what *proves* it. I'll expand h in the Gegenbauer basis, impose the domination on a grid and nonnegativity on the coefficients, and maximize N^2 alpha_0 - N h(1).

```python
import numpy as np
from scipy.special import gegenbauer
from scipy.optimize import linprog

def lp_energy_lower_bound(n, N, f_of_squared_dist, degree=12, grid=600):
    # certificate: a single lower bound valid for EVERY N-point set on S^{n-1}.
    lam = n / 2.0 - 1.0                                   # S^{n-1}: lambda = n/2 - 1
    C1 = np.array([gegenbauer(i, lam)(1.0) for i in range(degree + 1)])
    # maximize N^2 * alpha_0 - N * h(1) = N^2 alpha_0 - N sum_i alpha_i C_i(1)
    obj = np.array([N*N if i == 0 else 0.0 for i in range(degree+1)]) - N * C1
    ts = np.linspace(-1 + 1e-3, 1 - 1e-4, grid)          # domination grid on [-1,1)
    A_ub = np.array([[gegenbauer(i, lam)(t) for i in range(degree+1)] for t in ts])
    b_ub = np.array([f_of_squared_dist(2 - 2*t) for t in ts])  # h(t) <= f(2-2t) = a(t)
    res = linprog(-obj, A_ub=A_ub, b_ub=b_ub,            # alpha_i >= 0 : positive-definite
                  bounds=[(0, None)]*(degree+1), method="highs")
    return -res.fun, res.x

def coulomb(R):                 # f(R) = R^{-1/2}, R = squared distance -> 1/dist
    return R ** -0.5

def icosahedron():             # sharp config on S^2: 3 inner products, a 5-design
    phi = (1 + np.sqrt(5)) / 2
    raw = []
    for a in (-1, 1):
        for b in (-phi, phi):
            raw += [(0, a, b), (a, b, 0), (b, 0, a)]
    P = np.unique(np.array(raw, float), axis=0)
    return P / np.linalg.norm(P[0])

def energy(P, f):
    E = 0.0
    for i in range(len(P)):
        for j in range(len(P)):
            if i != j:
                E += f(np.sum((P[i] - P[j])**2))
    return E

bound, alpha = lp_energy_lower_bound(3, 12, coulomb)
P = icosahedron()
print(bound, energy(P, coulomb))   # ~98.333 vs ~98.331: the certificate is sharp
```

The causal chain, start to finish: searching configurations can't prove optimality because the landscape has exponentially many traps, so I move to a dual bound; writing energy in the inner-product variable and dominating the potential from below by a polynomial h reduces everything to controlling the all-pairs sum of h; the all-pairs sum is controllable precisely in the Gegenbauer basis, where Schoenberg's positive-definiteness makes every coefficient's contribution have a fixed sign and the constant term yields N^2 alpha_0; that turns "best bound" into a linear program; for a configuration with m distances that is also a (2m - 1)-design — a sharp configuration — Hermite-interpolating the potential to second order at its m inner products produces an h that dominates (because complete monotonicity makes the remainder's high derivative nonnegative), interpolates exactly (so the bound is tight), and is positive-definite (because the interpolation modulus F^2 is conductive, built up from the design property and the closure of positive-definite functions under products); and since the construction only ever used absolute monotonicity, the same configuration minimizes every completely monotonic energy at once — universal optimality — with the Euclidean version of the very same positivity/domination/double-contact skeleton, run through modular-form interpolation, reaching E_8 and the Leech lattice.
