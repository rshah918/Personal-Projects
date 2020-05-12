#find the sum of all the even terms in the Fibannaci sequence, such that the
#last number of the sequence does not exceed 4 million 


a, b = 1, 2
sum_ = 2

while a < 4000000:
    temp_value = a + b
    if temp_value % 2 == 0:
        sum_ = sum_ + temp_value

    a = b
    b = temp_value

print sum_
