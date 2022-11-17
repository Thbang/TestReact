from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import collections

rowList= []
numberList=[]
expectedList= []
remove4=[]
remove5=[]
remove6=[]
# s= input('번호를 입력하세요:')
NumRow = int(input("로또 구매 갯수 입력하세요:"))
for i in range(NumRow):
    remove4_1=[]
    remove4_2=[]
    remove4_3=[]
    remove5_1=[]
    remove5_2=[]
    lotteryNumber = list(map(int,input(str(i+1)+'번 째 추첨번호를 입력하세요:').split()))
    for v in lotteryNumber:
        if v not in numberList:
            numberList.append(v)

    for ii in range(5):
          
            if ii < 4:
                remove4_1.append(lotteryNumber[ii])
                remove4_2.append(lotteryNumber[ii+1])
                remove4_3.append(lotteryNumber[ii+2])   
            #     remove5_1.append(lotteryNumber[ii])
            #     remove5_2.append(lotteryNumber[ii+1])
            # elif ii>3:
            #     remove5_1.append(lotteryNumber[ii])
            #     remove5_2.append(lotteryNumber[ii+1])
       
               
    rowList.append(lotteryNumber)
    remove4.append(remove4_1)
    remove4.append(remove4_2)
    remove4.append(remove4_3)
    # remove5.append(remove5_1)
    # remove5.append(remove5_2)
    remove6.append(lotteryNumber)
numberList.sort()
# print(remove4_1)
# print(remove4_2)
# print(remove4_3)
# print(remove4)
# print(remove5_1)
# print(remove5_2)
# print(remove5)
# print(remove6)
expectedList = list(combinations(numberList, 6))

expectedList = [list(expectedList[x]) for x in range(len(expectedList))]
# print(expectedList) 


for ii  in expectedList[::]: 
    
    for v in range(int(len(remove4)/3)):
        vv = v * 3
        if ii[0] == remove4[vv][0] and ii[1] == remove4[vv][1] and ii[2] == remove4[vv][2] and ii[3] == remove4[vv][3]:
            expectedList.remove(ii)
        elif ii[1] == remove4[vv+1][0] and ii[2] == remove4[vv+1][1] and ii[3] == remove4[vv+1][2] and ii[4] == remove4[vv+1][3]:
            expectedList.remove(ii)
        elif ii[2] == remove4[vv+2][0] and ii[3] == remove4[vv+2][1] and ii[4] == remove4[vv+2][2] and ii[5] == remove4[vv+2][3]:
            expectedList.remove(ii) 
# print('길이')         
# print(len(expectedList))   
    
# numberList = [list(numberList[x]) for x in range(len(numberList))]
print("\n총"+str(len(numberList))+"종류 숫자가 있으며 리스트는 다음과 같습니다: \n"+str(numberList))
print("\n"+str(len(expectedList))+"개의 예측번호를 추측 할 수 있습니다.")





menuWhile = 0 
while menuWhile == 0 :
    print("\n---------Menu----------\n")
    print("1.당첨번호로 확률 조회\n")
    print("2.예상 예측번호 리스트 출력\n")
    print("3.예상 예측번호 리스트 내 값 존재 확인\n")
    print("4.종료\n")
    yn=input("메뉴 번호를 입력하세요:")
    if yn.lower() == '1':
        winningNumber = list(map(int,input('\n당첨번호를 입력하세요:').split()))
        if winningNumber in expectedList:
            print("예측번호 안에 당첨번호가 "+str(expectedList.count(winningNumber)) +"개 존재하며 확률은"+str(round(((1/len(expectedList))*100),3))+"%입니다.")
        else:
            print("조회 할 수 없는 당첨번호 입니다.\n")
            
    elif yn.lower() == '2':
        # dataset =pd.read_excel('./Lotto.xlsx', skiprows=[1], usecols=[1,2,3,4,5,6])
        dataset =pd.read_excel('./Lotto.xlsx', usecols=[1,2,3,4,5,6])
        # dataset =pd.read_excel('./Lotto.xlsx', usecols=[1,2,3,4,5,6])
        # print(dataset.values)
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
        
        print(dfNo1)
        print(dfNo2)
        print(dfNo3)
        print(dfNo4)
        print(dfNo5)
        print(dfNo6)
        
        
        def create_feature_table(data):
    
            feature_table=[]
            for i in range(len(data)):
                x= data[i]
        
                features={}
                features["len"] = len(x)
                features["mean"] = np.mean(x)
                features["std"] = np.std(x)
                features["max"] = np.max(x)
                features["min"] = np.min(x)
                features["max_idx"] = np.argmax(x)
                features["min_idx"] = np.argmin(x)
                features["skewness"] = stats.skew(x)
                features["kurtosis"] = stats.kurtosis(x)
                features["rms"] = np.sqrt(np.mean(np.square(x)))
                features["p2p"] = features["max"] - features["min"]
                features["crest_factor"] = features["max"] / features["rms"]
                feature_table.append(features)
            return pd.DataFrame(feature_table)
        
        # print(create_feature_table(dataset.values))
        # print(create_feature_table(expectedList))
        dfDataset = create_feature_table(dataset.values)
        dfExpectedList = create_feature_table(expectedList)



        X_train, X_test = train_test_split(dfDataset.values, test_size=0.2, random_state= 10)
        # list_df = pd.DataFrame(expectedList)
        list_df = dfExpectedList.values
        
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        list_df = scaler.transform(list_df)
        
        
        
        # clf = IsolationForest(n_estimators=200, random_state= 100, contamination=0.02)
        # clf.fit(X_train)
        # aa= clf.predict(X_train).tolist()
        # bb= clf.predict(X_test).tolist()
        # cc= clf.predict(list_df ).tolist()
        
        # print(len(X_train))
        # print(aa.count(1))
        # print(len(X_test))
        # print(bb.count(1))
        # print(len(list_df))
        # print(cc.count(1))
        
        
        oc_svm = OneClassSVM(kernel="rbf", nu=0.1, gamma= 0.0002)
        oc_svm.fit(X_train)
        aa= oc_svm.predict(X_train).tolist()
        bb= oc_svm.predict(X_test).tolist()
        cc= oc_svm.predict(list_df ).tolist()
        
        # print(len(X_train))
        # print(aa.count(1))
        # print(len(X_test))
        # print(bb.count(1))
        # print(len(list_df))
        # print(cc.count(1))
        # print(cc)
        # print(expectedList)
        result= []
        
        for i in range(len(cc)):
            if cc[i] == 1:
                result.append(expectedList[i])
         
        # print(result)
        print("당첨 있냐!!")
        print(str(result.count([8,16,26,29,31,36])))
        # print(str(result.count([6,7,9,11,17,18])))
        print("당첨 있냐??")
        
        
        
        
        
        
        total = []

        # for i in range(len(result)):
        for i in result[::]:
            temp=[]
            for v in range(len(dfNo1.values.tolist())):
                if dfNo1.values.tolist()[v][0] == i[0]:
                    temp.append(dfNo1.values.tolist()[v][2])
            for v in range(len(dfNo2.values.tolist())):
                if dfNo2.values.tolist()[v][0] == i[1]:
                    temp.append(dfNo2.values.tolist()[v][2])
            for v in range(len(dfNo3.values.tolist())):
                if dfNo3.values.tolist()[v][0] == i[2]:
                    temp.append(dfNo3.values.tolist()[v][2])
            for v in range(len(dfNo4.values.tolist())):
                if dfNo4.values.tolist()[v][0] == i[3]:
                    temp.append(dfNo4.values.tolist()[v][2])
            for v in range(len(dfNo5.values.tolist())):
                if dfNo5.values.tolist()[v][0] == i[4]:
                    temp.append(dfNo5.values.tolist()[v][2])  
            for v in range(len(dfNo6.values.tolist())):
                if dfNo6.values.tolist()[v][0] == i[5]:
                    temp.append(dfNo6.values.tolist()[v][2])                      
             
            # total.append(np.sum(temp)) 
            if len(temp) == 6:
                total.append(np.sum(temp))
            else:
                if [8,16,26,29,31,36] == i:
                    print("에에? 삭제 데쇼?")
                print(i)    
                result.remove(i)
        print("당첨 있냐!!")
        print(len(result))
        print(str(result.count([8,16,26,29,31,36])))
        # print(str(result.count([6,7,9,11,17,18])))
        print("당첨 있냐??")   
        col_name = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
        list_df = pd.DataFrame(result, columns=col_name)
        # print(list_df.head())    
        list_df['total']= total
        print(list_df.sort_values('total', ascending= False))
        print("예측 리스트에 당첨 있냐!!")
        temp11 = list_df.values.tolist()
        print(str(temp11.count([8,16,26,29,31,36,140])))
        # print(str(result.count([6,7,9,11,17,18])))
        print("당첨 있냐??")
        print(colNo1)
        print(colNo2)
        print(colNo3)
        print(colNo4)
        print(colNo5)
        print(colNo6)
        print("예상 데이터 개수:"+str(len(result)))
        print("평균: "+str(list_df.mean()))
        # # row 생략 없이 출력
        # pd.set_option('display.max_rows', None)
        # # col 생략 없이 출력
        # pd.set_option('display.max_columns', None)
        # random = list_df[(list_df['total']>=130)&(list_df['total']<180)].sort_values('total', ascending= False).sample(n=50)
        random = list_df[(list_df['total']>=140)&(list_df['total']<170)].sort_values('total', ascending= False)
        print(len(random))
        
        print("당첨 있냐!!")
        temp22 = list_df.values.tolist()
        print(str(temp22.count([8,16,26,29,31,36,140])))
        # print(str(temp22.count([6,7,9,11,17,18])))
        print("당첨 있냐??")
        
    elif yn == '3':
        expectedNumber = list(map(int,input('\n예측 번호를 입력하세요:').split()))
        temp =random[['번호1','번호2','번호3','번호4','번호5','번호6']].values.tolist()
        print(temp)
        print(len(temp))
        if expectedNumber in temp:
            print("예측번호 안에 예측번호가 "+str(temp.count(expectedNumber)) +"개 존재하며 확률은"+str(round(((1/len(temp))*100),3))+"%입니다.")
        else:
            print("조회 할 수 없는 예측번호 입니다.\n")
        # for i in list_df.tolist():
        # menuWhile = 1
        
                 
        
        
    elif yn == '4':
        print("종료")
        menuWhile = 1
    


# print(np.shape(expectedList))
# print(rowList)

