# TIER: greedy
# The "obvious" recipe: implement a classic, aesthetically-pleasing fractal L-system tree
# (the textbook ~22 degree branching angle, moderate length ratio, no leaf shrink) and
# never look at the instance's cost coefficients or sun schedule. More leaf material
# (taper=1.0, i.e. every tip keeps full area) is treated as strictly better -- it never
# is, once the crown gets deep or the biomass cost bites, because the fixed narrow angle
# packs tips too close together and they shade one another.
import sys

def main():
    sys.stdin.read()  # a "greedy" coder doesn't bother reading cost/sun specifics
    print("0.7 22.0 1.0 0.0")

if __name__ == "__main__":
    main()
