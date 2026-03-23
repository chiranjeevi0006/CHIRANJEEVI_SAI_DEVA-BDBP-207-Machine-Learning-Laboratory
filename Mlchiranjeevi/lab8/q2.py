def ordinal_encoding(data,order):
    ans=[]
    for x in data:
        ans.append(order[x])
    return ans

data = ["blue","red","green","yellow","purple","blue","yellow"]

order={
    "blue":0,
    "red":1,
    "green":2,
    "yellow":3,
    "purple":4
}
result=ordinal_encoding(data,order)
# print(result)

def one_hot_encoding(data):
    ans=[]
    categories = list(set(data))
    print(categories)
    for x in data:
        row = []
        for y in categories:
            if x ==y:
                row.append(1)
            else:
                row.append(0)
        ans.append(row)
    return ans
# data = ["blue","red","green","yellow","purple","blue","yellow"]
# result=one_hot_encoding(data)

print(result)
def main():
    data=["blue","red","green","yellow","purple"]
    print(ordinal_encoding(data,order))
    print(one_hot_encoding(data))
if __name__ == "__main__":
    main()
