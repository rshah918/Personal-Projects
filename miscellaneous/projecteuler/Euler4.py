#A palindrome reads the same both ways
# find the largest palindrome that is the product of 2 3-digit numbers



palindrome = 0
for i in range(100,999):
    for j in range(100,999):
        digits = [int(k) for k in str(i * j)]
        if digits == digits[::-1]:
            if i*j > palindrome:
                palindrome = i*j

print palindrome
