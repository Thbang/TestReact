import numpy as np
import pandas as pd
col_name = ['colname1', 'colname2', 'colname3']
list1 = [[1, 2, 3 ,4 ,5 ,6], [7, 8, 9,10 , 11, 12]]
df = pd.DataFrame(list1, columns=col_name)


col_name2 = ['idx', 'score']
list2 = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60]]
df2 = pd.DataFrame(list2, columns=col_name2)

print(df)
print(df2)

total = []

for i in range(len(list1)):
    temp=[]
    for v in range(len(list2)):
        if list2[v][0] == list1[i][0]:
            temp.append(list2[v][1])
    for v in range(len(list2)):
        if list2[v][0] == list1[i][1]:
            temp.append(list2[v][1])
    for v in range(len(list2)):
        if list2[v][0] == list1[i][2]:
            temp.append(list2[v][1])
    print(np.sum(temp))
    total.append(np.sum(temp))               

print(total)
df['total']= total
print(df)
print(df.sort_values('total', ascending= False))
            

