# Fiat-Shamir Heuristic

For a public-coin three-message proof with transcript `(a, c, z)` and verifier predicate `V(x, a, c, z)`, replace the verifier's random challenge with a transcript-derived value:

```text
c = H(domain, x, a)
proof = (a, z)
accept iff V(x, a, c, z) = 1
```

For a signature derived from an identification protocol, bind the message into the challenge:

```text
c = H(domain, public_key, message, a)
signature = (a, z)
accept iff V(public_key, a, c, z) = 1
```

In the original Fiat-Shamir modular-square-root scheme, the identity-derived public values are `u_j = f(I, j)` and the card stores secrets with `s_j^2 * u_j = 1 mod n`. The signer first computes commitments `z_i = r_i^2 mod n`, then uses the first `k*t` bits of `f(m, z_1, ..., z_t)` as the verifier challenge matrix `e_ij`, computes responses `y_i = r_i * product_{j:e_ij=1} s_j mod n`, and sends the identity, message, challenge matrix, and responses. Verification reconstructs each `z_i = y_i^2 * product_{j:e_ij=1} u_j mod n`, then checks that `f(m, z_1, ..., z_t)` reproduces the same challenge matrix.

The method is not a hash appended to a proof. The hash or random function occupies the verifier's challenge role. Its input must include the fixed commitment and all relevant public context, including the message for signatures, so the prover cannot choose a convenient challenge first and fit a commitment afterward.
