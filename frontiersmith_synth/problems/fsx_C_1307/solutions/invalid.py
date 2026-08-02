# TIER: invalid
# Malformed answer: wrong length (missing the last step) -- must score 0.
import sys, json

def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    print(json.dumps({"cull": [0.05] * (T - 1)}))

if __name__ == "__main__":
    main()
