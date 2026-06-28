import re, sys
base="/srv/home/bohanlyu/innovation_proior/data_v4/fcs-p2-03"
sol=open(f"{base}/verify/sol.cpp").read()
def extract(md):
    t=open(md).read()
    blocks=re.findall(r"```cpp\n(.*?)```", t, re.S)
    return blocks[-1] if blocks else None
for f in ["reasoning.md","train_answer.md"]:
    b=extract(f"{base}/{f}")
    ok = (b==sol)
    print(f"{f}: cpp block == sol.cpp -> {ok}")
    if not ok and b is not None:
        # show first diff
        for i,(x,y) in enumerate(zip(b,sol)):
            if x!=y:
                print(f"  first diff at char {i}: md={x!r} sol={y!r}")
                print(f"  md len={len(b)} sol len={len(sol)}")
                break
        else:
            print(f"  prefix match; md len={len(b)} sol len={len(sol)}")
