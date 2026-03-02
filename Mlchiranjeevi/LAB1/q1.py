A=[[1,2,3],[4,5,6]]
transpose_d=[]

for i in range(len(A[0])):
    row = []
    for j in range(len(A)):
        row.append(A[j][i])
    transpose_d.append(row)

print(transpose_d)