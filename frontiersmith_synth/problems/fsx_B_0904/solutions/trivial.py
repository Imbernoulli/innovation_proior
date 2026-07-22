# TIER: trivial
import sys, json

def main():
    _ = json.load(sys.stdin)  # ignore everything -- constant blind setback
    print(json.dumps({"level": 3}))

main()
