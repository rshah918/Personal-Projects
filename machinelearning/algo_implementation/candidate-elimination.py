#implement candidate elimination model for machine learning
#For first training vector:
    #If an element in vector does not match vector label, mark it as "to-be-negated"
        #A flag can be added into the GBS indicating whether it should be negated.
#For all subsequent vectors:
    #negate all elements in the "to-be-negated" list. If element does not match label, delete from GBS


#INPUT: vector containing subvectors. last element in subvector corrosponds to output
train_set = [[0,0,1,0],[1,0,0,1], [0,1,0,1], [1,1,1,0]]
def negate(num_to_negate):
    #negates binary number
    negated_num = abs(num_to_negate-1)
    return negated_num
#Start with specific boundary set (SBS) == 0 and general Boundary set (GBS) == 1
specific_boundary_set = [[0,0]] #value is negation flag; "1" == negate, key is index of point in training_sample
general_boundary_set = {0:0, 1:0, 2:0} #value is negation flag; "1" == negate, key is index of point in training_sample

for i in train_set:
    for j in i:
        sample_label = i[-1]
        #Special case for first training_sample
        if train_set.index(i) == 0:
            #If an element doesnt match the sample label, add negation flag to the element in the general_boundary_set
            if (j != sample_label):
                general_boundary_set[i.index(j)] = 1

        else: #for all other training samples
            #if value in general_boundary_set == 1, negate the element before comparing to vector label
            if general_boundary_set.get(i.index(j)) == 1:
                negated_num = negate(general_boundary_set[i.index(j)])
                if negated_num != sample_label:
                    #delete key value pair. If it is already deleted, pass
                    try:
                        del general_boundary_set[i.index(j)]
                    except:
                        pass
            #if value == 0 just compare without negation
            else:
                if (j != sample_label):
                    #delete key value pair. If it is already deleted, pass
                    try:
                        del general_boundary_set[i.index(j)]
                    except:
                        pass

print general_boundary_set


#TODO: Find out what the SBS is used for
