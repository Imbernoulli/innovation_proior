# TIER: trivial
# Never cull. The predator stays overabundant and prey stays chronically
# suppressed -- a mediocre, non-catastrophic score with no effort at all.
import sys, json

def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    print(json.dumps({"cull": [0.0] * T}))

if __name__ == "__main__":
    main()
