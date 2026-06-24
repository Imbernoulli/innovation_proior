import sys
from itertools import product

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    s = data[1] if n > 0 else ""
    # Exhaustively enumerate every assignment of ON/OFF to the n lamps.
    # A pattern is valid iff:
    #   - no broken lamp ('x') is ON, and
    #   - no two adjacent lamps are both ON.
    count = 0
    for bits in product([0, 1], repeat=n):
        ok = True
        for i in range(n):
            if bits[i] == 1 and s[i] == 'x':
                ok = False
                break
            if i > 0 and bits[i] == 1 and bits[i - 1] == 1:
                ok = False
                break
        if ok:
            count += 1
    print(count)

main()
