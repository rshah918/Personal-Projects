def factorial(n):
    if (n == 1 or n == 0):
        return 1
    product = n * factorial(n-1)
    return product

print(factorial(6))
