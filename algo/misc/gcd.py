
v1 = 14
v2 = 49

# gcd - greatest common divisor
def gcd(a, b):
    while b:
        a %= b
        a, b = b, a
    return a

# recursive
def gcd_r(a, b):
    if not b:
        return a
    else:
        return gcd_r(b, a % b)

print(gcd_r(v1, v2))

# least common multiple = num1 * num2 / gcd
lcm = v1 * v2 / gcd_r(v1, v2)
print(lcm)

