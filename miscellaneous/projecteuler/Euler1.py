multiples = []
def option1():
    global multiples
    for i in range(1000):
        if i % 3 == 0 or i % 5 == 0:
            multiples.append(i)

    print sum(multiples)

option1()

#step1: find the multiples of 3 and 5 that are below 1000
#step2: add it togethr
