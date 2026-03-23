import numpy as np

x= np.array([
    [1,0,2],
    [0,1,1],
    [2,1,0],
    [1,1,1],
    [0,2,1]
])

ans=np.cov(x,rowvar=False)
print(ans)