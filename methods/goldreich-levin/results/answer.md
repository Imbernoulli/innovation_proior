# Goldreich-Levin

For any strong one-way function \(f\), define

\[
g(x,r)=(f(x),r), \qquad |r|=|x|.
\]

The predicate

\[
b(x,r)=\langle x,r\rangle \bmod 2
\]

is hard-core for \(g\). Given \(f(x)\) and a uniformly random \(r\), no probabilistic polynomial-time algorithm can predict \(b(x,r)\) with non-negligible advantage over \(1/2\).

The reduction is the substance of the theorem. If a predictor has non-negligible advantage \(\epsilon\), then for a noticeable set of inputs \(x\) it is correlated with the Hadamard codeword \(r\mapsto \langle x,r\rangle\). For such an \(x\), choose \(m=\operatorname{poly}(n,1/\epsilon)\) and \(\ell=\lceil\log_2(m+1)\rceil\), sample masks \(s_1,\ldots,s_\ell\), guess their labels, derive labels for the nonempty subset-xor masks \(r_J\), and recover each coordinate by majority votes of

\[
\alpha_J \oplus G(f(x),r_J\oplus e_i).
\]

When the seed-label guesses are right, the identity

\[
\langle x,r_J\oplus e_i\rangle=\langle x,r_J\rangle\oplus x_i
\]

turns each predictor call into a vote for \(x_i\). The \(r_J\)'s are uniform and pairwise independent, so enough of them give concentration for the majority vote. Enumerating the seed-label assignments yields a polynomial-size candidate list, and testing candidates with \(f\) identifies the preimage when it is present.

Thus a one-bit prediction advantage becomes full preimage recovery. The random inner product is hard-core because any efficient predictor for it would violate the assumed one-wayness of \(f\).
