# Goal: two candidate windows whose exact averages are extremely close, and whose
# cross-multiplication product exceeds int64 so a long-long comparison picks the wrong one.
# Window B: length m, sum = m*V + 1  (average slightly above V).
# Window A: length 1, value = V       (average exactly V).
# True winner = B (avg V + 1/m > V). Cross products: A.num*B.den = V*m ; B.num*A.den = m*V+1.
# These are ~10^9 * 1.8e5 ~ 1.8e14 -> fits int64. Need bigger.
#
# Instead make BOTH windows long so products blow up:
# A: length p, sum sA ;  B: length q, sum sB.  Compare sA*q vs sB*p.
# Want sA*q ~ 4e19 (> 9.2e18) and sA/p vs sB/q extremely close so int64 wrap flips it.
# Pick p=q=180000. sA = 10^9*180000 ~ 1.8e14. sA*q ~ 3.24e19  -> overflows int64. Good.
# Make A and B both windows of length 180000 of all +1e9 except B has one element 1 less,
# so B.avg < A.avg by a hair. The wrong int64 comparison may pick B.
#
# Build: first 180000 values = 1e9 (window A = [0,180000)).
# Then make a second region whose best window has average just below A's so the
# correct answer is A's average exactly 1e9. If int64 overflow flips it, broken picks the lower one.
n = 200000
L = 180000
vals = [10**9]*n
# perturb a middle stretch downward slightly so some long window has avg 1e9 - tiny,
# but never above 1e9. The unique max average is 1e9/1 (any all-1e9 window).
# To create overflow-sensitive *near-ties*, lower exactly one element far right by 1.
vals[n-1] = 10**9 - 1
print(n, L)
print(' '.join(map(str, vals)))
