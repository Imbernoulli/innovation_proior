import itertools, subprocess, sys
sol = "/srv/home/bohanlyu/innovation_proior/data_v4/fcs-p2-03/verify/sol"
def brute(a):
    best=None
    for i in range(len(a)):
        p=1
        for j in range(i,len(a)):
            p*=a[j]
            if best is None or p>best: best=p
    return best
mis=0; tot=0
# exhaustive over n<=4, values in -3..3
vals=range(-3,4)
for n in range(1,5):
    for a in itertools.product(vals,repeat=n):
        inp=f"{n}\n{' '.join(map(str,a))}\n"
        out=int(subprocess.run([sol],input=inp,capture_output=True,text=True).stdout.strip())
        exp=brute(list(a))
        tot+=1
        if out!=exp:
            mis+=1
            print("MISMATCH",a,"sol",out,"brute",exp)
            if mis>=10: sys.exit(1)
print(f"exhaustive tot={tot} mismatches={mis}")
