#What is the smallest positive number that is evenly divisible by all of the numbers
#from 1 to 20?


#generate a list of multiples for every number
#smallest number that is present in all lists of multiples is the solution to the problem.


#dont need a list of multiples of 3 because 6, 9, 12, 18, and 24 all share a common list.
#focus on genearting a list of multiples for the larger numbers and the prime numbers in the list of 1-20

# figure out which numbers are prime in the list besides 1, 2, 3
#generate a list of multiples for each of those numbers
#iterate over every element of that list, nest if statements to find a common value
#the first common value should also be the smallest, as the list of multiples are in ascending order

'''divisor = range(1,21)
test_vals = range(2,divisor[-1])
critical_values = []
tracker = []'''
#below block finds all prime numbers in the list of divisors and adds it to "critical values"
'''for i in divisor:
    for j in test_vals:
        if j != i:
            if i % j == 0:
                tracker.append('orange peel')
    if len(tracker) == 0:

        critical_values.append(i)
    else:
        del tracker[::]'''
#below block generates a list containing all of the multiples that we need to test
'''list_of_multiples = []
for prime in critical_values:
    n = 10000
    while n < 100001:
        list_of_multiples.append(prime * n)
        n = n + 1

print list_of_multiples'''

#finds the common multiple, and that is the answer
#below code takes a long time to exceucute, mainly because the list of multiples is so long
#modify code so that it scans for a common multiple as the list is being generated, then delete the list every failed iteration so that the size of the list is smaller
#oh wait, the program will still need to iterate over the same number of numbers, so it wont help the speed
# study more about the multiples of primes, see if there is a pattern that I can use to eliminate
# the number of multiples i need to iterate over
'''for val1 in list_of_multiples:
    common_multiple = []
    for val2 in list_of_multiples:
        if val2 == val1:
            common_multiple.append(val2)
            print common_multiple
    if len(common_multiple) == len(critical_values):
         print common_multiple[0]
         break'''

#UPDATE OCT-5-2017: i am overcomplicating things
a = '--------------------------------------------------------------------------'
#do this tomorrow
divisor = [11, 13, 14, 16, 17, 18, 19, 20]
temp_num = 0
factor = None
primes = []


def factor_gen():
    global temp_num
    global divisor
    for i in range(2, 999999999, 20):
        if all(i % x == 0 for x in divisor):
            print i
            break


factor_gen()
# answer is 232792560!!!
