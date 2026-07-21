Two numbers in the constraints fix the whole shape of this before I write a line. `N` runs to `10^18`, so I cannot touch terms one at a time — anything `O(N)` per query is dead on arrival and I need something logarithmic in `N`. And `p` runs to `2*10^9`, so a single product of two reduced residues reaches about `(2*10^9)^2 = 4*10^18`, past the signed 64-bit ceiling; every modular multiply therefore needs a 128-bit intermediate. So the skeleton is settled already: `unsigned long long` state, `__uint128_t` for the products, and an advance-by-powers algorithm rather than an advance-by-steps one. Everything else follows from those two facts.

The task itself: `f(0)=a`, `f(1)=b`, `f(i)=c*f(i-1)+d*f(i-2)` for `i>=2`, and each query wants `S(N) = f(0)+f(1)+...+f(N-1) mod p`, the sum of the first `N` terms. The coefficients `a,b,c,d` can be negative in `[-10^9,10^9]`, which means every one of them has to be reduced into `[0, MOD)` before it enters any product, or the printed residue could come out negative.

The tempting wrong turns are two, and both die on the same constraint. One is to enumerate: generate `f(2), f(3), ...` up to index `N-1` and accumulate mod `p`. It is trivially correct — I will in fact reuse exactly this as an offline oracle for small `N` — but at `O(N)` per query a single `N=10^18` query needs `10^18` iterations, decades of wall time; no constant factor rescues that, and a precomputed table sized to even `10^7` is still eleven orders of magnitude short of `10^18`. The other is to lean on a closed form: for ordinary Fibonacci the prefix sum is `f(N+1)-1`, and the small cases (`N=0->0`, `N=1->a`, `N=2->a+b`) are tidy. But `f(N+1)-1` holds only for the `c=d=1` family; it is simply false for `c=d=0`, for `d=0`, for `c=0`, and for a generic coefficient triple — and those are precisely the degeneracies the evaluation stresses. There is no per-family shortcut I can safely memorize across arbitrary `(a,b,c,d,p)`. One general `O(log N)` algorithm has to cover everything.

The recurrence is linear, so the pair `(f(i), f(i-1))` advances by the fixed companion matrix `[[c,d],[1,0]]`, which gives the n-th *term* in `O(log N)`. But I need the *sum*, not the term. The way to get it at the same cost is to augment the state with the running prefix sum and find one linear map that advances term and sum together. Define, for `i>=1`,

```
v_i = [ f(i), f(i-1), P(i) ]^T,   where P(i) = f(0)+f(1)+...+f(i) = S(i+1).
```

I want `v_{i+1} = M v_i`. Writing each new component in the old ones:

- `f(i+1)   = c*f(i) + d*f(i-1)`  — the recurrence.
- `f(i)     = f(i)`               — shift down.
- `P(i+1)   = P(i) + f(i+1) = c*f(i) + d*f(i-1) + P(i)`  — the sum picks up the new term.

So, with columns ordered `[f(i), f(i-1), P(i)]`,

```
        [ c  d  0 ]
  M  =  [ 1  0  0 ]
        [ c  d  1 ]
```

The top two rows are the standard companion block; the third row is the augmentation — it is row 1 plus a `1` in the `P` column, i.e. "add the freshly computed `f(i+1)` to the carried sum." That third row is the entire point — it makes one matrix power emit the sum, not merely the term.

I anchor at `i=1`, where everything is known: `v_1 = [ f(1), f(0), P(1) ]^T = [ b, a, a+b ]^T`. Applying `M` exactly `k` times reaches `v_{1+k}`. Since `S(N)` is the sum of indices `0..N-1`, that is `P(N-1)`, so I need `v_{N-1} = M^{N-2} v_1` and read off its third component. This is valid for `N>=2`; the exponent `N-2` is where I have to be careful not to underflow, so `N=0` and `N=1` come out by hand — `S(0)=0`, `S(1)=f(0)=a` — before the matrix path runs. And `p=1` needs its own short-circuit ahead of any matrix being built — the modular algebra below is what forces that ordering.

The Fibonacci sample fixes the off-by-one count. With `a=b=c=d=1`, `N=10`, `k = N-2 = 8`: walking the third component forward from `P(1)=2` gives `+2->4, +3->7, +5->12, +8->20, +13->33, +21->54, +34->88, +55->143`, i.e. `P(9)=143=S(10)`, matching the expected `143`. So the exponent is `N-2` and the anchor `v_1` is placed right.

The one genuinely problem-specific numeric trap is `p=1`. Under `MOD=1` every residue must be `0`, but a literal `1` written into the matrices — the identity diagonal, `M.m[1][0]`, `M.m[2][2]` — is *not* reduced mod 1, so it carries a stray `1` that leaks a nonzero residue exactly when `matpow` returns the identity (exponent 0) or when such a one survives a multiply. Two things fix it and I take both: a guard `if (MOD==1) { print 0; continue; }` placed before any matrix is built, so the degenerate modulus never reaches the linear algebra; and, defensively, writing `1 % MOD` wherever a literal one enters a matrix, so the algebra stays correct even if the guard were ever moved. The guard-first ordering is the real fix; the `1 % MOD` is belt-and-suspenders.

Now the overflow accounting. In the `3x3` multiply the inner accumulator adds three `mulmod` results, each `< MOD < 2*10^9`; unchecked it could reach `~6*10^9`, still under the `u64` ceiling but I keep it provably small by subtracting `MOD` after each add, so the running sum stays `< 2*MOD < 4*10^9`, nowhere near `2^63`. The product inside `mulmod` is `(u128)a*b <= 4*10^18`, far under `2^128`; a 64-bit product would have overflowed once `p` passed `~3*10^9` and silently corrupted the answer. The 128-bit intermediate is exactly what makes `p` up to `2*10^9` safe.

The degenerate families are where a half-remembered closed form would quietly be wrong, so I check that the *single* general matrix gets them with no special-casing. For `c=d=0`, `f(i)=0` for `i>=2`: row 0 of `M` becomes `[0,0,0]` and row 2 becomes `[0,0,1]`, so `P` never grows past `P(1)=a+b` — correct, `S(N)=a+b` for `N>=2`. For `d=0, c=1`, `f(i)=f(i-1)` is constant `=b` from index 1, so `S(N)=a+(N-1)*b`; the `P`-accumulation produces this on its own, and the sample's second query (`2 3 1 0`, `N=7`, `p=11`) has terms `2,3,3,3,3,3,3`, sum `20`, `20 mod 11 = 9`. For `c=0`, `f(i)=d*f(i-2)` splits into independent even/odd chains, but the augmented map iterates blind to parity and handles it. Each of these is a family where `f(N+1)-1` is false, and each is covered by the same matrix with no special-casing.

I differential-test the code against the term-by-term brute across the deterministic edge bundle (`p=1`, `N=0/1/2`, negatives with a small modulus, Fibonacci, `c=d=0`, `d=0`, `c=0`, coefficients near `±10^9`) and several hundred random small-`N` instances over a spread of prime and composite moduli, with no mismatch — which also confirms the negative-input reduction lands every coefficient in `[0,MOD)`. The brute cannot reach `N=10^18`, so the large-`N` path is checked separately against an independent matrix reference written straight from the state definition and itself validated against the brute on small `N`, over `N` in `[10^17,10^18]` and moduli up to `2*10^9`; they agree. The cost is comfortable: each query is one `3x3` power, at most ~60 squarings plus ~60 conditional multiplies, each `27` 128-bit products, so even `10^5` worst-case `N=10^18` queries run well inside the 2-second limit.
