The canonical name for this result is the Nash embedding theorem, or simply Nash embedding.  It is the statement that every Riemannian manifold can be realized isometrically as a submanifold of some Euclidean space.  The theorem is famous for the statement, but its lasting influence comes from the method: it reframes geometric realization as a flexible nonlinear PDE problem and then solves that PDE by a carefully designed iteration rather than by an explicit formula.

The first step is to write the geometric condition in coordinates.  If M is an m-dimensional manifold with Riemannian metric g and f : M -> R^N is a candidate embedding, then f is isometric exactly when the pullback of the Euclidean metric e equals g.  In local coordinates this becomes the first-order nonlinear system g_ij(x) = sum_alpha partial_i f^alpha(x) partial_j f^alpha(x).  The unknowns are the N coordinate functions of f, and the data is the metric tensor g.  The key observation is that when N is large, there are far more unknown functions than metric equations.  On an m-manifold the metric has m(m+1)/2 independent components, while the embedding supplies N coordinate functions.  This makes the system highly underdetermined, and Nash's insight was to treat that underdetermination as analytic room rather than as mere embedding dimension.

Before Nash, the natural approach was to try to construct f explicitly or to use the implicit function theorem on the operator P(f) = f^*e.  Linearizing around an approximate embedding gives a first-order PDE for a correction u, and in principle one could solve that linearized equation and iterate.  The difficulty is derivative loss.  The operator P depends on first derivatives of f, and its linearization produces a system whose solution requires more regularity than the data naturally provides.  In a fixed Banach space the inverse is unbounded, so a standard Newton iteration cannot be repeated: after one correction the next correction needs smoothness that has already been spent.  Classical explicit geometry has the opposite limitation.  It can build special surfaces or local analytic patches, but it offers no general global mechanism for arbitrary Riemannian metrics.

Nash solved both problems with two related ideas.  For C^1 isometric embeddings, he introduced what is now called the Nash twist.  Start with a short embedding, meaning one whose induced metric is strictly smaller than g.  The gap g - f^*e is a positive definite metric defect.  Nash adds to the embedding rapidly oscillating perturbations, usually in normal directions.  Because the oscillations are high frequency, the perturbation itself is small in the C^0 sense, so the map stays close to the original one.  But the derivative of the perturbation is large, and its quadratic contribution to the induced metric has a controllable average.  Each corrugation therefore adds a prescribed piece to the metric while barely moving the underlying point set.  Iterating this process with finer and finer frequencies drives the metric defect to zero and yields a C^1 isometric embedding.

The geometric intuition is that a tiny wrinkle can carry a lot of derivative energy.  The map looks almost unchanged to the eye, yet its derivative has been adjusted so that the induced metric matches the target.  This is why the C^1 theorem can produce wild, non-rigid embeddings that classical differential geometers would not have expected.  The rigidity of isometric embedding in low codimension is replaced by flexibility once enough normal directions are available.

For smooth embeddings, Nash used a second idea that became the prototype of the Nash-Moser implicit function theorem.  Instead of a plain Newton iteration, he inserted a smoothing step into the correction loop.  Each iteration solves the linearized equation as before, which loses derivatives, but then the result is smoothed before it is added to the approximate solution.  Smoothing introduces its own error, but the fast quadratic contraction of the Newton step dominates that error if the smoothing is chosen appropriately.  The iteration therefore converges in a scale of Banach spaces rather than in a single space.  This is the analytical mechanism that overcomes derivative loss and produces a smooth, or even analytic, isometric embedding.

The breakthrough is not merely that every Riemannian manifold fits inside Euclidean space.  It is that geometric existence can be obtained as the limit of controlled corrections.  Nash embedding changed what counts as a solution to a geometric problem.  Instead of deriving the embedded shape from intrinsic curvature formulas, one designs an iterative procedure that spends the extra degrees of freedom in high-dimensional Euclidean space.  The C^1 version spends them through high-frequency geometry, and the smooth version spends them through smoothing and Newton contraction.  Both versions replace explicit construction with convergent analysis.

The influence of the method extends well beyond embedding theory.  The Nash-Moser iteration is now a standard tool for nonlinear PDEs with derivative loss, appearing in KAM theory, micro local analysis, and geometric PDE.  The Nash twist inspired the theory of convex integration and has been used to construct surprising solutions to equations in fluid dynamics and geometry.  In machine learning, the same heuristic of using high-frequency perturbations to satisfy constraints while staying close in a coarse norm appears in various regularization and generative modeling contexts.  Thus Nash embedding is both a classical theorem and a template for turning underdetermined nonlinear problems into convergent approximation schemes.

The following Python script illustrates the C^1 Nash twist in the simplest possible setting, a one-dimensional curve in high-dimensional Euclidean space.  The target is a prescribed speed squared along the curve.  Starting from a short straight embedding whose speed is too small, the script adds many independent high-frequency sinusoidal corrugations in orthogonal normal directions.  Each corrugation contributes a small prescribed amount to the speed squared, and their combined quadratic contributions approximate the target metric.  Using many normal directions is the key feature that lets the oscillations cancel in the average while their squared derivatives add up correctly.

```python
import numpy as np

def nash_twist_curve(target_speed_sq, t, n_dirs=200, base_freq=16, seed=0):
    rng = np.random.default_rng(seed)
    scale = 0.9  # short initial speed, so scale**2 < min(target_speed_sq)
    y = np.zeros((n_dirs, len(t)))
    dy = np.zeros((n_dirs, len(t)))
    for i in range(n_dirs):
        freq = base_freq + i
        # Each normal direction carries an equal share of the metric defect.
        share = (target_speed_sq - scale ** 2) / n_dirs
        amplitude = np.sqrt(2.0 * share) / (2.0 * np.pi * freq)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        y[i] = amplitude * np.sin(2.0 * np.pi * freq * t + phase)
        dy[i] = amplitude * (2.0 * np.pi * freq) * np.cos(2.0 * np.pi * freq * t + phase)
    speed_sq = scale ** 2 + np.sum(dy ** 2, axis=0)
    return scale * t, y, speed_sq

if __name__ == "__main__":
    t = np.linspace(0.0, 1.0, 20001)
    # A varying target speed squared that stays above the initial short speed squared.
    target = 1.5 + 0.5 * np.sin(2.0 * np.pi * t) ** 2
    x, y, speed_sq = nash_twist_curve(target, t)
    print(f"Initial speed^2: {0.9 ** 2:.4f}")
    print(f"Mean target speed^2: {np.mean(target):.4f}")
    print(f"Mean final speed^2: {np.mean(speed_sq):.4f}")
    print(f"Max |defect|: {np.max(np.abs(target - speed_sq)):.4f}")
    print(f"Mean |defect|: {np.mean(np.abs(target - speed_sq)):.4f}")
    print(f"Max |y| over all normal directions: {np.max(np.abs(y)):.6f}")
```

This script is only a cartoon of Nash's full construction.  A real Nash embedding must handle an arbitrary Riemannian metric on a manifold of any dimension, ensure that the map remains an embedding rather than merely an immersion, and control the limit in C^1 or smooth topology.  Nevertheless, the code captures the central mechanism: high-frequency normal corrugations add derivative energy to the metric while keeping the map itself nearly fixed.  That mechanism is the heart of Nash embedding and the reason the theorem opened a new chapter in nonlinear PDE and differential geometry.