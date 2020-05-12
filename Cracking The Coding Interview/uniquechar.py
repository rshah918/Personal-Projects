#Implement an algo to determine if a string has all unique chars

def uniqueChars(s):
    #O(n)
    s = list(s)
    char_list = list(range(255))
    for char in s:
        if char_list[ord(char)] == True:
            return False
        else:
            char_list[ord(char)] = True

    return True;



print(uniqueChars("ABCD"))


def hashTableUniqueChars(s):
    #O(n)
    s = list(s)
    hashset = {}
    for char in s:
        try:
            if hashset[char] == 1:
                return False
        except:
            hashset[char] = 1
    return True

print(hashTableUniqueChars("ABCDD"))
