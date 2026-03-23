import matplot.pyplot as plt

x1=[]
Y=[]
start = -10
stop =10
num = 100
step = (stop - start)/(num-1)

for i in range(num):
    x=start+i*step
    y=2*(x**2)+3*x+4
    x1.append(x)
    Y.append(y)

plt.plot(x1,Y)
plt.show()