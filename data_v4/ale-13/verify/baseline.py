# Trivial baseline: concatenate all fragments in given order (always feasible).
import sys
with open(sys.argv[1]) as f:
    lines=f.read().split("\n")
i=0
while lines[i].strip()=="": i+=1
n=int(lines[i].split()[0]); i+=1
frags=[]
while i<len(lines) and len(frags)<n:
    frags.append(lines[i].rstrip("\r")); i+=1
sys.stdout.write("".join(frags)+"\n")
