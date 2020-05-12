
def multiply(a,b):
    if b == 0:
        return 0
    if b == 1:
        return a
    if(b > 0):
        product = a + multiply(a, b-1)
    

    return product


print(multiply(2,20))
