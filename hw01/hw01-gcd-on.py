def gcd(a, b):
    a, b = min(a, b), max(a, b)
    for i in range(a, 0, -1):
        if a % i == 0 and b % i == 0:
            return i