

'''
  pop off an element, add it to the new subset
'''

def subsetgenerator(subset):
    #base case: do nothing
    if(len(subset) == 0):
        return [[]]

    first = subset[0]
    ss = subsetgenerator(subset[1:])
    new = []
    for s in ss:
        new.append([first]+s)

    subset = new+ss
    return subset


subset = subsetgenerator(["a","b","c"])
print(subset)
