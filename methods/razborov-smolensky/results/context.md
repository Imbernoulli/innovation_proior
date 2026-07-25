## Problem location
The Razborov-Smolensky method is the next step in the 1980s line of small-depth circuit lower bounds: after Ajtai/Furst-Saxe-Sipser/Yao/Hastad used random restrictions to prove that `PARITY` is not in `AC0`, it extends the target to `AC0[p]`, circuits with mod-counting gates. The core question is: can a constant-depth, polynomial-size circuit that allows `MOD_p` gates compute the counting function `MOD_q` for a different modulus?

## Circuit model
`AC0[p]` denotes constant-depth, polynomial-size, unbounded fan-in circuits of `AND`, `OR`, `NOT`, `MOD_p` gates, where `p` is usually taken to be prime. The typical Razborov-Smolensky result is: if `p` and `q` are distinct primes, then `MOD_q` is not in `AC0[p]`; more quantitatively, an `AC0[p]` circuit of depth `d` computing `MOD_q` needs an exponential-size lower bound, commonly stated in lecture notes as `2^{Omega(n^{1/(2d)})}`.
