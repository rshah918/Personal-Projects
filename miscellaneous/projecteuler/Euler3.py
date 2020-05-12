#find the largest factor of a number


# find the prime factors of a number
#find the largest of those prime factors
#if the rightmost digit is a 2, 5, or a 0 then it is for sure not prime


input_ =  2100


temp_num = input_
factor = None
primes = []
def factor_gen():
    global temp_num
    global input_
    global primes
    #keeps divides the input_ by numbers that go up by one until it evenly divides
    for i in range(2, input_):
        if i != temp_num:
            if temp_num % i == 0:
                temp_num = temp_num / i
                #pass the now smaller number through the function until we get a prime number
                return factor_gen()

    primes.append(temp_num)

    p = 1
    for n in primes:
        p = n * p

    temp_num = input_ / p

    while p != input_:
        return factor_gen()
    print primes

    maximum = 0
    for j in primes:
        if j > maximum:
            maximum = j
    print maximum
factor_gen()
