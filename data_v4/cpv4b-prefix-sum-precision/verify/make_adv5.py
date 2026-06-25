# Few-window construction to force the overflow flip into the FINAL fraction comparison.
# n large, L = n-1. Candidate windows:
#   W_full = [0,n)            length n,   sum = T
#   W_left = [0,n-1)          length n-1, sum = T - a[n-1]
#   W_right= [1,n)            length n-1, sum = T - a[0]
# Make W_right the TRUE max average, but with int64 overflow the comparison flips
# so a long-long solution would output W_full or W_left instead.
#
# Use the flip from find_flip3: products near 2^63 with length ~ n.
# Pick n=200000, so length n-1=199999 and n=200000. We compare W_right(s1,199999) vs others.
# We want s_right/199999 to be the max, with cross products overflowing.
#
# Simplest: make all elements equal to V=10**9 EXCEPT a[0] smaller, so dropping a[0]
# (W_right) raises the average above the full window. Then W_right is the unique max.
#   a[0] = X (< V), a[1..n-1] = V.
#   avg_full = ((n-1)V + X)/n
#   avg_right = V  (window [1,n) all V)  -> highest, true max = V/1 = V exactly.
#   avg_left = (a[0..n-2]) = ((n-2)V + X)/(n-1)  < V.
# True answer: V/1 = 10^9/1. Both int128 and int64 should agree (V exact). Not a flip.
#
# To force a FLIP we need a NEAR-TIE between two long windows. Make a[0] and a[n-1] tuned so
# avg_left and avg_right are within 1/(n*(n-1)) of each other AND products overflow.
V=10**9
n=200000
# Let a[1..n-2] = V (the common middle, n-2 elements). a[0]=p, a[n-1]=q.
# W_left=[0,n-1): sum = p + (n-2)V ; length n-1
# W_right=[1,n): sum = (n-2)V + q ; length n-1
# Equal length => comparison reduces to comparing sums p+(n-2)V vs (n-2)V+q i.e. p vs q.
# Same length cancels overflow advantage. Need DIFFERENT lengths to exploit overflow.
#
# Different lengths: L=1, but then short windows dominate. Constrain via making ALL prefix
# averages of short windows low, and the max achieved by a specific LONG window vs another
# LONG window of different length. That's exactly two long windows -> need n>=~2*100000.
#
# Use L=99000. Windows length in [99000,200000]. Engineer array as two halves again but
# ensure NO straddling window beats the two intended ones by making the boundary a deep valley.
half=100000
sA=92233720368547   # avg ~9.2233e8, length 100000
sB=92233720368548   # avg slightly higher, length 100000  (TRUE max)
def realize(total,length,lo=-10**9,hi=10**9):
    q,r=divmod(total,length); vals=[q]*length
    for i in range(r): vals[i]+=1
    assert sum(vals)==total and all(lo<=v<=hi for v in vals)
    return vals
left=realize(sA,half)
right=realize(sB,half)
# Make the join a valley: the last element of left and first of right are already ~9.2e8.
# A straddling window mixing them averages between the two ~9.2e8 values, all < sB/half.
# Since both halves average ~9.2e8 and right is the max, and any straddling window is a
# weighted average of segments each <= max-segment-average, the max over length>=L is
# the all-right window (length 100000) = sB/100000. Set L=100000 so only full-half windows
# of length>=100000 qualify; straddling ones of length 100000 average < sB/100000.
vals=left+right
print(2*half, half)  # n=200000, L=100000
print(' '.join(map(str,vals)))
