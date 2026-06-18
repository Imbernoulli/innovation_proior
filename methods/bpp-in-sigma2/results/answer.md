# BPP in the Second Level

Let `L in BPP`. By amplification, choose a deterministic polynomial-time predicate `M(x,r)` whose final number of random bits is `m=poly(|x|)` and whose error is at most `1/(3m)`. Thus

- if `x in L`, then `Pr_r[M(x,r)=1] >= 1 - 1/(3m)`;
- if `x notin L`, then `Pr_r[M(x,r)=1] <= 1/(3m)`.

For fixed `x`, set `A_x = { r in {0,1}^m : M(x,r)=1 }` and use XOR as addition on `{0,1}^m`. For a shift `y`, write `A_x + y = { a xor y : a in A_x }`.

The alternating test is

`exists y_1,...,y_m in {0,1}^m forall z in {0,1}^m, OR_{i=1}^m M(x, z xor y_i)=1`.

The witness is a list of `m` shifts, so its length is `m^2`. The verifier runs `M` on the shifted strings `z xor y_i` and ORs the answers, which is deterministic polynomial time. The XOR sign is correct because `z in A_x + y_i` iff `z xor y_i in A_x`.

If `x in L`, the complement of `A_x` has density at most `1/(3m)`. Independent random shifts miss a fixed point `z` with probability at most `(1/(3m))^m`; union bounding over all `2^m` points gives miss probability at most `(2/(3m))^m < 1`. Hence some shift list covers the whole cube.

If `x notin L`, then `|A_x| <= 2^m/(3m)`. Any `m` shifted copies have union size at most `m|A_x| <= 2^m/3`, so some `z` is outside all of them and the universal check fails.

Thus `L in Sigma_2^P`. Since `BPP` is closed under complement, `BPP subseteq Sigma_2^P cap Pi_2^P`.
