import sys
data=sys.stdin.read().split()
n=int(data[0]); a=[int(x) for x in data[1:1+n]]
i,j=0,n-1; turn=0; tot=[0,0]
while i<=j:
    if a[i]>=a[j]:
        tot[turn]+=a[i]; i+=1
    else:
        tot[turn]+=a[j]; j-=1
    turn^=1
print(tot[0])
