a=[5,6,-3,3,-5]
# plain kadane window
n=len(a)
# whole-array prefix sums to understand
# plain max subarray:
best=None;cur=0;cl=0;bl=br=0
for r in range(n):
    if cur<=0: cur=a[r];cl=r
    else: cur+=a[r]
    if best is None or cur>best: best=cur;bl=cl;br=r
print("plain max subarray sum=",best,"window=",a[bl:br+1],"idx",bl,br)
print("min in window=",min(a[bl:br+1]),"greedy delete-most-neg=",best-min(a[bl:br+1]))
# optimal: try whole subarrays with one deletion
B=float('-inf');info=None
for l in range(n):
    s=0
    for r in range(l,n):
        s+=a[r]
        if s>B: B=s;info=(l,r,None,s)
        if r-l+1>=2:
            for k in range(l,r+1):
                if s-a[k]>B: B=s-a[k];info=(l,r,k,s-a[k])
print("optimal=",B,"l,r,deleted_idx,val=",info)
