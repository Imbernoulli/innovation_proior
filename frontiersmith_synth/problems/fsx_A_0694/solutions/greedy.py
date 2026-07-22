# TIER: greedy
# Round-robin: cycle through all K skills in fixed index order (0,1,...,K-1,
# 0,1,...), giving everyone an equal share of the montage. This "looks safe" --
# nobody is neglected -- but it is completely oblivious to the interference
# matrix (antagonist pairs get drilled back-to-back every single lap, eroding
# each other right after being trained) and to diminishing returns (skills that
# saturated early keep getting re-drilled at zero marginal value while the
# montage's last slot goes to whichever skill happens to land on T-1 mod K, not
# to whichever skill actually needs the protection of the final refresher).
import sys, json


def main():
    inst = json.load(sys.stdin)
    K = inst["K"]
    T = inst["T"]
    seq = [t % K for t in range(T)]
    print(json.dumps({"sequence": seq}))


if __name__ == "__main__":
    main()
