# TIER: trivial
# Route every job to unit 0, ignoring capacity, fatigue, and repair entirely.
import sys, json

def main():
    inst = json.load(sys.stdin)
    m = len(inst["jobs"])
    print(json.dumps({"assignment": [0] * m}))

if __name__ == "__main__":
    main()
