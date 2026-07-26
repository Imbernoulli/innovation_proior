# TIER: invalid
# Emits a k x k matrix with two identical rows -- correct shape and range, but
# singular mod p for every p, so it can never be a valid change-of-basis. The
# checker's invertibility gate must reject this with Ratio: 0.0.
import sys


def main():
    data = sys.stdin.read().split()
    p = int(data[0]); k = int(data[1])
    rows = []
    for i in range(k):
        row = [0] * k
        row[0] = 1  # row 0 and row 1 (if k>=2) both become [1,0,0,...]
        rows.append(row)
    out = [" ".join(map(str, row)) for row in rows]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
