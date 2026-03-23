import numpy as np
# my_dataset = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# my_dataset = np.array(my_dataset)
# print(my_dataset)
# print(type(my_dataset))
# my_dataset=my_dataset * 2
# print(my_dataset)
#
# my_dataset=my_dataset % 2
# print(my_dataset)
#
# array=np.array([[["A","B"],["C","D"],["E","F"],["G","H"]],
#                 [["A","B"],["C","D"],["E","F"],["G","H"]],
#                 [["A","B"],["C","D"],["E","F"],["G","H"]]])
# print(array[2][3][1]) #chain indexing
#
# word= array[0,1,1] + array[2,0,0] + array[1,3,1]
# print(word)

array = np.array([[1,2,3,4],
                  [5,6,7,8],
                  [9,10,11,12],
                  [13,14,15,16]])

#arry start:end:step
# print(array[2:, 2:])

# scalar arthmetic

aray=np.array([1,2,3])

# print(aray **5)
# print(aray +1)
# print(aray -1)
# print(aray /1)
# print(aray *5)
# print(np.sqrt(array))
# print(np.round(aray))
#
# radii = np.array([1,2,3,4,5])
#
# print(np.pi * radii **2)

# elemet-wiser arthimetic\

# array1=np.array([1,2,3,4,5])
# array2=np.array([5,6,7,8,9])
#
# print(array1 +array2)
# print(array1-array2)
# print(array1 / array2)
# print(array1 ** array2)

# comparison operators

# scores =np.array([1,2,3,4,5])
#
# print(scores<3)
#
# scores[scores <3] =0
# print(scores)

array2=np.array([[5,6,7,8]])
array3=np.array([[1],[2],[3],[4]])

print(array3.shape)
print(array2.shape)

print(array2 * array3)