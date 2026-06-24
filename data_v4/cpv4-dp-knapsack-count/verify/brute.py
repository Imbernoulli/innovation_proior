import sys

MOD = 1000000007

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    B = int(data[idx]); idx += 1
    J = int(data[idx]); idx += 1
    price = []
    joy = []
    for _ in range(n):
        price.append(int(data[idx])); idx += 1
        joy.append(int(data[idx])); idx += 1

    # Exhaustively enumerate every one of the 2^n subsets, check the two
    # constraints exactly, and count. Obviously correct; exponential, only for
    # small n.
    count = 0
    for mask in range(1 << n):
        tot_price = 0
        tot_joy = 0
        for i in range(n):
            if mask & (1 << i):
                tot_price += price[i]
                tot_joy += joy[i]
        if tot_price == B and tot_joy >= J:
            count += 1
    print(count % MOD)

if __name__ == "__main__":
    main()
