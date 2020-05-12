#find the 10001 prime number

#use a while loop, until the length of the list of prime numbers reaches 100001


primes = [2]

n = 3
while len(primes) < 100001:
    test = 'prime'
    test_vals = range(2,n-1) 
    for i in test_vals:
        if n % i == 0:
            test = 'not prime'
    if test == 'prime':
        primes.append(n)
    n = n + 1

    print len(primes)
print "the 100,001 prime is", primes[-1]
