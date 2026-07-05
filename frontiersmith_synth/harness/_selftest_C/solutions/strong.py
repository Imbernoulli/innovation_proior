# TIER: strong
import sys
d=sys.stdin.read().split(); it=iter(d); n=int(next(it)); s=[]
for i in range(n):
    p=int(next(it)); c=int(next(it))
    if c<p: s.append(i+1)
print(len(s)); print(*s)
