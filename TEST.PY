# lst1 = [['A', 'B', 'C', 'D'],['E', 'F', 'G', 'H']]
# lst2 = ['C', 'D']

# # intersection = list(set(lst1) & set(lst2))
# del lst1[0]
# for a in lst1:
#     lst
# print(lst1)

import collections
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import IsolationForest

# dataset =pd.read_excel('./Lotto.xlsx', skiprows=[1], usecols=[1,2,3,4,5,6], skipfooter=900)+
dataset =pd.read_excel('./Lotto.xlsx',  usecols=[1,2,3,4,5,6], skipfooter=500)
No1 = dataset['번호1'].tolist()
No2 = dataset['번호2'].tolist()
No3 = dataset['번호3'].tolist()
No4 = dataset['번호4'].tolist()
No5 = dataset['번호5'].tolist()
No6 = dataset['번호6'].tolist()

colNo1 = collections.Counter(No1)
colNo2 = collections.Counter(No2)
colNo3 = collections.Counter(No3)
colNo4 = collections.Counter(No4)
colNo5 = collections.Counter(No5)
colNo6 = collections.Counter(No6)
print(colNo1)
print(colNo2)
print(colNo3)
print(colNo4)
print(colNo5)
print(colNo6)

dfNo1 = pd.DataFrame.from_dict(colNo1, orient='index').reset_index().sort_values(by= 0,ascending= False)
a = []
for i in range(len(dfNo1)):
    a.append(i+1)
a.sort(reverse=True)
dfNo1['score'] = a
dfNo2 = pd.DataFrame.from_dict(colNo2, orient='index').reset_index().sort_values(by= 0,ascending= False)
a = []
for i in range(len(dfNo2)):
    a.append(i+1)
a.sort(reverse=True)
dfNo2['score'] = a
dfNo3 = pd.DataFrame.from_dict(colNo3, orient='index').reset_index().sort_values(by= 0,ascending= False)
a = []
for i in range(len(dfNo3)):
    a.append(i+1)
a.sort(reverse=True)
dfNo3['score'] = a
dfNo4 = pd.DataFrame.from_dict(colNo4, orient='index').reset_index().sort_values(by= 0,ascending= False)
a = []
for i in range(len(dfNo4)):
    a.append(i+1)
a.sort(reverse=True)
dfNo4['score'] = a
dfNo5 = pd.DataFrame.from_dict(colNo5, orient='index').reset_index().sort_values(by= 0,ascending= False)
a = []
for i in range(len(dfNo5)):
    a.append(i+1)
a.sort(reverse=True)
dfNo5['score'] = a
dfNo6 = pd.DataFrame.from_dict(colNo6, orient='index').reset_index().sort_values(by= 0,ascending= False)
a = []
for i in range(len(dfNo6)):
    a.append(i+1)
a.sort(reverse=True)
dfNo6['score'] = a
dfNo1.values.tolist()
print(dfNo1)
print(dfNo2)
print(dfNo3)
print(dfNo4)
print(dfNo5)
print(dfNo6)


col_name = ['colname1', 'colname2', 'colname3', 'colname4', 'colname5', 'colname6']
list1 = [[1, 2, 3 ,4 ,5 ,6], [7, 8, 9,10 , 11, 12]]
# list1 = [[2, 8 , 19,22,32,42]]
df = pd.DataFrame(list1, columns=col_name)


total = []

# for i in range(len(list1)):
for i in  list1[::]:
    print("i출력")
    print(i)
    temp=[]
    for v in range(len(dfNo1.values.tolist())):
        if dfNo1.values.tolist()[v][0] == i[0]:
            print("1번이당")
            print(dfNo1.values.tolist()[v][2])
            temp.append(dfNo1.values.tolist()[v][2])
    for v in range(len(dfNo2.values.tolist())):
        if dfNo2.values.tolist()[v][0] == i[1]:
            print("2번이당")
            print(dfNo2.values.tolist()[v][2])
            temp.append(dfNo2.values.tolist()[v][2])
    for v in range(len(dfNo3.values.tolist())):
        if dfNo3.values.tolist()[v][0] == i[2]:
            print("3번이당")
            print(dfNo3.values.tolist()[v][2])
            temp.append(dfNo3.values.tolist()[v][2])
    for v in range(len(dfNo4.values.tolist())):
        if dfNo4.values.tolist()[v][0] == i[3]:
            print("4번이당")
            print(dfNo4.values.tolist()[v][2])
            temp.append(dfNo4.values.tolist()[v][2])
    for v in range(len(dfNo5.values.tolist())):
        if dfNo5.values.tolist()[v][0] == i[4]:
            print("5번이당")
            print(dfNo5.values.tolist()[v][2])
            temp.append(dfNo5.values.tolist()[v][2])  
    for v in range(len(dfNo6.values.tolist())):
        if dfNo6.values.tolist()[v][0] == i[5]:
            print("6번이당")
            print(dfNo6.values.tolist()[v][2])
            temp.append(dfNo6.values.tolist()[v][2]) 
                
    print(temp)                 
    print(np.sum(temp))
    print(len(temp))
    # total.append(np.sum(temp))
    if len(temp) == 6:
        total.append(np.sum(temp))
    else:
        list1.remove(i)               

print(list1)







# print(dfNo2)
# print(dfNo3)
# print(dfNo4)
# print(dfNo5)
# print(dfNo6)

# X_train, X_test = train_test_split(dataset.values, test_size=0.1, random_state= 10)


# X_outliers = [[12, 22, 31, 35, 40, 44],[11, 21, 30, 34, 39, 43],[6, 15, 16, 23, 22, 43],[6, 8, 10, 12, 14, 16]]
# list_df = pd.DataFrame(X_outliers)
# # print(list_df.values)
# clf = IsolationForest(max_samples=800, random_state= 10, contamination=0.05)
# clf.fit(X_train)
# aa =clf.predict(X_test).tolist()
# bb=clf.predict(list_df.values)
# cc=  clf.predict(X_train).tolist()
# print(len(X_train))
# print(cc.count(1))
# print(len(X_test))
# print(aa.count(1))
# print(bb)

# from scipy import stats
# def create_feature_table(data):
    
#     feature_table=[]
#     for i in range(len(data)):
#         x= data[i]
        
#         features={}
#         features["len"] = len(x)
#         features["mean"] = np.mean(x)
#         features["std"] = np.std(x)
#         features["max"] = np.max(x)
#         features["min"] = np.min(x)
#         features["max_idx"] = np.argmax(x)
#         features["min_idx"] = np.argmin(x)
#         features["skewness"] = stats.skew(x)
#         features["kurtosis"] = stats.kurtosis(x)
#         features["rms"] = np.sqrt(np.mean(np.square(x)))
#         features["p2p"] = features["max"] - features["min"]
#         features["crest_factor"] = features["max"] / features["rms"]
#         feature_table.append(features)
#     return pd.DataFrame(feature_table)

