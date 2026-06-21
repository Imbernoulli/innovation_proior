## Research question

Finite-dimensional linear algebra says that a Hermitian matrix can be understood by an orthonormal basis of eigenvectors. In that setting the operator is a weighted sum of orthogonal projections,

`A = sum_j lambda_j P_j`,

and any reasonable function of the operator is obtained by applying the function to the eigenvalues:

`f(A) = sum_j f(lambda_j) P_j`.

The functional-analytic question is what remains of this picture for a bounded operator on an infinite-dimensional Hilbert space. A bounded self-adjoint operator can have no eigenvectors at all while still having a large spectrum. The goal is therefore to replace eigenvectors by spectral subspaces, replace sums by integrals, and replace a diagonal matrix by a measure of orthogonal projections.

For a complex Hilbert space `H` and a bounded normal operator `N`, the desired object is a rule assigning to each Borel set `B` in the spectrum a projection `E(B)` onto the part of `H` whose spectral values lie in `B`. The operator should then be recovered as

`N = int_{sigma(N)} z dE(z)`,

and every bounded Borel function should have a canonical meaning:

`f(N) = int_{sigma(N)} f(z) dE(z)`.

## Background

The finite-dimensional theorem works because normal matrices are unitarily diagonalizable. The diagonal entries are eigenvalues, and the eigenspaces give mutually orthogonal projections. This already contains the durable pattern: the important object is not a list of chosen eigenvectors, but the projection onto each eigenspace.

Compact self-adjoint operators on Hilbert space still behave much like matrices. Their nonzero spectrum consists of eigenvalues with finite-dimensional eigenspaces, accumulating only at zero, so an eigen-expansion remains meaningful after allowing a countable sum and a possible null space.

On `L^2([0,1])`, the operator `(Mf)(t)=t f(t)` is bounded and self-adjoint. It has no nonzero eigenvector: `(t-lambda)f(t)=0` forces `f` to be supported on a single point, hence `f=0` in `L^2`. For a Borel set `B subset [0,1]`, the projection is multiplication by `1_B`, and the operator is multiplication by the coordinate function.

This example points to the right abstraction. Hilbert space geometry turns each projection `E(B)` into scalar measures by

`mu_x(B)=<E(B)x,x>`,

and more generally `mu_{x,y}(B)=<E(B)x,y>` by polarization. Integration against these scalar measures defines operator integrals. Orthogonality of spectral subspaces is encoded by multiplication of indicators:

`E(B)E(C)=E(B cap C)`.

Normality is the algebraic hypothesis that makes the calculus commutative. A bounded normal operator generates a commutative C*-algebra, so polynomials in `N` and `N*` extend to continuous functions on `sigma(N)`. The remaining step is to pass from continuous functions to Borel functions so that discontinuous cuts such as `1_B` become projections.

## Baselines

- **Eigenbasis diagonalization.** Exact for finite-dimensional normal operators. The diagonal entries are eigenvalues and the eigenspaces give mutually orthogonal projections.

- **Compact self-adjoint expansion.** For compact self-adjoint operators, the nonzero spectrum consists of eigenvalues with finite-dimensional eigenspaces, and the operator is a countable sum of scalar multiples of the corresponding eigenspace projections.

- **Resolvent-only spectral analysis.** The resolvent `(N-zI)^{-1}` is defined for `z` outside the spectrum and detects analytic structure and spectral location.

- **Continuous functional calculus.** The map `f -> f(N)` for `f in C(sigma(N))` is a canonical unital `*`-homomorphism. It contains polynomials, square roots of positive operators, and uniform limits.

- **Multiplication-operator representation.** An operator can be represented as multiplication by a measurable function on an appropriate `L^2` space, giving an intuitive diagonal form.

## Evaluation settings

This is a theorem, so success is logical rather than empirical.

- **Analytic setting.** Complex Hilbert spaces and bounded normal operators; bounded self-adjoint operators as the real-spectrum special case.

- **Spectral setting.** Compact subsets of `C` for normal operators, compact subsets of `R` for self-adjoint operators, and Borel sets in those spectra.

- **Stress cases.** Operators with continuous spectrum and no eigenvectors; operators with mixed point and continuous spectrum; compact operators where the theorem must reduce to the familiar countable eigenprojection expansion.

- **Calculus setting.** Continuous functions should agree with the usual continuous functional calculus, characteristic functions should produce orthogonal projections, and bounded Borel functions should produce bounded operators with `||f(N)|| <= ||f||_infty`.

- **Standard of success.** The assignment `B -> E(B)` must be a countably additive projection-valued measure, the operator must be recovered by integrating the coordinate function, and the construction must be unique.

## Code framework

The artifact is a theorem and proof. The reusable proof scaffold is:

1. Start with the continuous functional calculus for a bounded normal operator.
2. Convert vector states `<f(N)x,y>` into scalar measures on the spectrum.
3. Extend the calculus from continuous functions to bounded Borel functions.
4. Define spectral projections as `E(B)=1_B(N)`.
5. Prove projection, orthogonality, strong countable additivity, and uniqueness.
6. Recover the operator as an integral of the coordinate function and check that the finite atomic case collapses to ordinary diagonalization.
