def wrap(x):
    x &= (2**64 - 1)
    if x >= 2**63: x -= 2**64
    return x
base=10**9
# We want one product just below 2^63 region and the other just above so wrap flips order.
# 2^63 = 9223372036854775808 ~ 9.223e18.
# Choose products P_A = sA*dB and P_B = sB*dA. If P_A slightly < 2^63 (stays positive)
# and P_B slightly > 2^63 (wraps to large negative), then int64 sees P_A > P_B even if
# exact P_A < P_B -> flip when exact says B wins but int64 says A wins.
# Construct: dA=dB=d. Then factor d common; product = d*sA vs d*sB. Want d*sB ~ 2^63.
# 2^63 / 1e9 = 9.22e9. So sB ~ 9.22e9 / d... pick d=200000 -> sB ~ 46116... let's solve.
# d * sB ≈ 2^63 -> sB ≈ 2^63/d.
import math
TWO63 = 2**63
for d in [200000, 199999, 150000, 100000]:
    sB = TWO63 // d            # makes d*sB just under 2^63
    sB += 1                    # push d*sB just over 2^63 -> wraps negative
    sA = sB - 1                # avgA < avgB (since same d, smaller sum). B is TRUE winner.
    # feasibility: sums must be reachable with d elements each in [-1e9,1e9]: |s|<=1e9*d
    if abs(sA) > base*d or abs(sB) > base*d:
        print(f"d={d}: sums {sA},{sB} infeasible (cap {base*d})"); continue
    PA = sA*d; PB = sB*d
    exact_B_wins = PB > PA
    l=wrap(PA); r=wrap(PB)
    int64_B_wins = r > l
    print(f"d={d} sA={sA} sB={sB}  exact_B_wins={exact_B_wins} int64_B_wins={int64_B_wins}  FLIP={exact_B_wins!=int64_B_wins}")
