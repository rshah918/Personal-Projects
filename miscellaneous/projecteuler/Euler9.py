
for a in range(100000):
    for b in range(100000):
        if ((a^2+b^2)^(1/2))==1000-a-b and (1000.00 - a - b).is_integer() == False:
            print a
            print b
            print ((1000 - a - b))
