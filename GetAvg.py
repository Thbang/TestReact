import collections
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# dataset =pd.read_excel('./Lotto.xlsx',  usecols=[1,2,3,4,5,6], skipfooter=500)
dataset =pd.read_excel('./Lotto.xlsx',  usecols=[1,2,3,4,5,6])
list1= dataset.values.tolist();
print(list1)
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
# print(colNo1)
# print(colNo2)
# print(colNo3)
# print(colNo4)
# print(colNo5)
# print(colNo6)

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
# print(dfNo1)
# print(dfNo2)
# print(dfNo3)
# print(dfNo4)
# print(dfNo5)
# print(dfNo6)

total = []
# for i in range(len(list1)):
for i in  list1[::]:
    # print("i출력")
    # print(i)
    temp=[]
    for v in range(len(dfNo1.values.tolist())):
        if dfNo1.values.tolist()[v][0] == i[0]:
            # print("1번이당")
            # print(dfNo1.values.tolist()[v][2])
            temp.append(dfNo1.values.tolist()[v][2])
    for v in range(len(dfNo2.values.tolist())):
        if dfNo2.values.tolist()[v][0] == i[1]:
            # print("2번이당")
            # print(dfNo2.values.tolist()[v][2])
            temp.append(dfNo2.values.tolist()[v][2])
    for v in range(len(dfNo3.values.tolist())):
        if dfNo3.values.tolist()[v][0] == i[2]:
            # print("3번이당")
            # print(dfNo3.values.tolist()[v][2])
            temp.append(dfNo3.values.tolist()[v][2])
    for v in range(len(dfNo4.values.tolist())):
        if dfNo4.values.tolist()[v][0] == i[3]:
            # print("4번이당")
            # print(dfNo4.values.tolist()[v][2])
            temp.append(dfNo4.values.tolist()[v][2])
    for v in range(len(dfNo5.values.tolist())):
        if dfNo5.values.tolist()[v][0] == i[4]:
            # print("5번이당")
            # print(dfNo5.values.tolist()[v][2])
            temp.append(dfNo5.values.tolist()[v][2])  
    for v in range(len(dfNo6.values.tolist())):
        if dfNo6.values.tolist()[v][0] == i[5]:
            # print("6번이당")
            # print(dfNo6.values.tolist()[v][2])
            temp.append(dfNo6.values.tolist()[v][2]) 
                
    # print(temp)                 
    # print(np.sum(temp))
    # print(len(temp))
    # total.append(np.sum(temp))
    if len(temp) == 6:
        total.append(np.sum(temp))
    else:
        list1.remove(i)      
df = pd.DataFrame(total, columns = ['sum'])
print(df)              
print(len(total))
print(total)
print(max(total))
print(min(total))
print(np.mean(total))

# plt.hist(total)


# sns.distplot(df['sum'])

# 


# df.sort_index(ascending=False).reset_index(drop=True)

# scaler = MinMaxScaler()
# scale_cols = ['sum']
# df_scaled = scaler.fit_transform(df[scale_cols])
# df_scaled = pd.DataFrame(df_scaled)
# df_scaled.columns = scale_cols

# print(df_scaled)

train_data=pd.DataFrame(df.loc[:700,['sum']])
test_data=pd.DataFrame(df.loc[700:,['sum']])
#분리된 데이터 시각화
ax = train_data.plot()
test_data.plot(ax=ax)
# plt.legend(['train', 'test'])

# plt.show()

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler() 
train_data_sc=scaler.fit_transform(train_data)
test_data_sc= scaler.transform(test_data)


#학습 데이터와 테스트 데이터( ndarray)를 데이터프레임으로 변형한다.
train_sc_df = pd.DataFrame(train_data_sc, columns=['Scaled'], index=train_data.index)
test_sc_df = pd.DataFrame(test_data_sc, columns=['Scaled'], index=test_data.index)


#학습 데이터와 테스트 데이터( ndarray)를 데이터프레임으로 변형한다.
train_sc_df = pd.DataFrame(train_data_sc, columns=['Scaled'], index=train_data.index)
test_sc_df = pd.DataFrame(test_data_sc, columns=['Scaled'], index=test_data.index)
#LSTM은 과거의 데이터를 기반으로 미래을 예측하는 모델이다. 따라서, 과거 데이터를 몇 개 사용해서 예측할 지 정해야 한다. 여기서는 30개(한 달)를 사용한다.  
for i in range(1, 501):
    train_sc_df ['Scaled_{}'.format(i)]=train_sc_df ['Scaled'].shift(i)
    test_sc_df ['Scaled_{}'.format(i)]=test_sc_df ['Scaled'].shift(i)

#nan 값이 있는 로우를 삭제하고 X값과 Y값을 생성한다.
x_train=train_sc_df.dropna().drop('Scaled', axis=1)
y_train=train_sc_df.dropna()[['Scaled']]

x_test=test_sc_df.dropna().drop('Scaled', axis=1)
y_test=test_sc_df.dropna()[['Scaled']]


#대부분의 기계학습 모델은 데이터프레임 대신 ndarray구조를 입력 값으로 사용한다.
#ndarray로 변환한다.
x_train=x_train.values
x_test=x_test.values

y_train=y_train.values
y_test=y_test.values

print(y_train.shape)


#LSTM 모델에 맞게 데이터 셋 변형
x_train_t = x_train.reshape(x_train.shape[0], 300,1)
x_test_t = x_test.reshape(x_test.shape[0], 300, 1)

from keras.layers import LSTM 
from keras.models import Sequential 
from keras.layers import Dense 
import keras.backend as K 
from keras.callbacks import EarlyStopping 

K.clear_session() 
# Sequeatial Model
model = Sequential() 
# 첫번째 LSTM 레이어
model.add(LSTM(30,return_sequences=True, input_shape=(300, 1))) 
# 두번째 LSTM 레이어
model.add(LSTM(42,return_sequences=False))  
# 예측값 1개
model.add(Dense(1, activation='linear')) 
# 손실함수 지정 - 예측 값과 실제 값의 차이를 계산한다. MSE가 사용된다. 
# 최적화기 지정 - 일반적으로 adam을 사용한다.
model.compile(loss='mean_squared_error', optimizer='adam') 
model.summary()


#손실 값(loss)를 모니터링해서 성능이 더이상 좋아지지 않으면 epoch를 중단한다.
#vervose=1은 화면에 출력
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)

#epochs는 훈련 반복 횟수를 지정하고 batch_size는 한 번 훈련할 때 입력되는 데이터 크기를 지정한다.
model.fit(x_train_t, y_train, epochs=50,
          batch_size=20, verbose=1, callbacks=[early_stop])

y_pred = model.predict(x_test_t)

#테스트의 Y값(실측값) 과 예측값을 비교한다.
t_df=test_sc_df.dropna()
y_test_df=pd.DataFrame(y_test, columns=['close'], index=t_df.index)
y_pred_df=pd.DataFrame(y_pred, columns=['close'], index=t_df.index)

ax1=y_test_df.plot()
y_pred_df.plot(ax=ax1)
plt.legend(['test','pred'])

plt.show()