#find the difference between the sum of the first 100 squares and the square of the sum

#create a variable set to a range, turn that into a list
#for loop, square each number and replace
#use sum()
#set variable equal to the sum of first 100, and square
#subtract

sum1 = 0
for i in range(1,11):
    square = i*i
    sum1 = sum1 + square

sum2 = 0
for j in range(1,101):
    sum2 = sum2 + j

answer = abs(sum1-(sum2*sum2))
print answer


#that was way too easy
