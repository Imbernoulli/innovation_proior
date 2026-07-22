#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of fsx_B_1010 to stdout.
Format:
  line 1: n k
  line 2: Dstar Lamstar
  line 3: wD wL
All 10 cases are fully determined by testId (no external randomness needed;
the difficulty/trap ladder is baked into the fixed table below, ascending in
grid resolution n^k). Cases 3,5,6,7,8,9,10 are TRAP cases: the target
lacunarity sits close to what a spatially-DISPERSED arrangement of the same
per-level cell count achieves, and far from what the natural
row-major/CLUSTERED arrangement (the obvious first construction) achieves --
so matching dimension alone (easy) does not matter get lacunarity right.
"""
import sys

# (n, k, Dstar, Lamstar, wD, wL)
TABLE = {
    1:  (3, 3, 1.4183, 2.6931, 2.0, 1.0),
    2:  (3, 4, 1.1909, 5.1352, 1.5, 1.0),
    3:  (4, 3, 1.2680, 5.1916, 2.0, 1.3),
    4:  (4, 4, 1.7098, 1.8465, 2.0, 1.3),
    5:  (5, 3, 1.2723, 6.9504, 1.5, 1.0),
    6:  (5, 4, 1.7090, 1.8837, 2.0, 1.5),
    7:  (6, 3, 1.5001, 4.0614, 2.0, 1.3),
    8:  (6, 4, 1.8322, 1.6207, 2.5, 1.0),
    9:  (3, 6, 1.1660, 7.5524, 3.0, 0.7),
    10: (4, 5, 1.4724, 6.0906, 2.0, 1.2),
}


def main():
    t = int(sys.argv[1])
    n, k, Dstar, Lamstar, wD, wL = TABLE[t]
    print(n, k)
    print(f"{Dstar:.4f} {Lamstar:.4f}")
    print(f"{wD:.4f} {wL:.4f}")


if __name__ == "__main__":
    main()
