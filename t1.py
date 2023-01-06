#1 Find S
import csv

dataset = csv.reader(open("Weather.csv","rt"))
dataset = list(dataset)
print(dataset)


attributes = ['Sky','Temp','Humidity','Wind','Water','Forecast']
target = ["Yes","Yes","No","Yes"]

hypothesis = ['0'] * len(attributes)


for i in range(len(target)):
    if(target[i] == 'Yes'):
        for j in range(len(attributes)):
            if(hypothesis[j]=='0'):
                hypothesis[j] = dataset[i][j]
            if(hypothesis[j]!= dataset[i][j]):
                hypothesis[j]='?'

    print(i+1,'=',hypothesis)

print("\nThe Maximally specific hypothesis for the training instance is ", hypothesis)

#2. Candidate Elimination

import pandas as pd
import numpy as np

data = pd.read_csv("Weather2.csv")
print(data)
concepts = np.array(data.iloc[:,0:-1])
target = np.array(data.iloc[:,-1])
print(concepts,end="\n")
print(target)

def learn(concepts,target):
    specific_h = concepts[0].copy()
    print(specific_h)

    general_h = [["?" for j in range(len(specific_h))] for i in range(len(specific_h))]
    print(general_h)
    
    for i, h in enumerate(concepts):
        
        if target[i] == "Yes":
            for x in range(len(specific_h)):
            
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'        
        if target[i] == "No":
            for x in range(len(specific_h)):
            
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("steps of Candidate Elimination Algorithm",i+1)
        print(specific_h)
        print(general_h)

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print("Final Specific_h:", s_final, sep="\n")
print("Final General_h:", g_final, sep="\n")

#3. KNN

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data=load_iris()
iris=load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)
model=KNeighborsClassifier(n_neighbors=10)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print('Accuraccy of KNN: \t',accuracy_score(y_pred,y_test))
pd.DataFrame({'Actutal': y_test, 'Prediction': y_pred,'Correct classification':(y_test==y_pred)})

#4. Naive Bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

dataset = pd.read_csv("naive.csv")

dataset_df = pd.DataFrame(dataset)

en = preprocessing.LabelEncoder()
dataset_df_encoded = dataset_df.apply(en.fit_transform)

data = dataset_df_encoded.drop(['play'], axis=1)
target = dataset_df_encoded['play']
print(data)
print(target)

X_train,X_test,Y_train,Y_test = train_test_split(data,target,test_size = 0.25)
model = GaussianNB()
learntModel = model.fit(X_train,Y_train)
prediction = learntModel.predict(X_test)
print("Output",list(prediction))
print(list(Y_test))

print("Accuracy : ", metrics.accuracy_score(prediction,Y_test))

#5. Locally Weighted Regression

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point,xmat, k):
    m,n= np.shape(xmat)
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
    return weights

def localWeight(point,xmat,ymat,k):
    wei = kernel(point,xmat,k)
    W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))
    return W

def localWeightRegression(xmat,ymat,k):
    m,n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)
    return ypred

data = pd.read_csv('tips_LinearRegression.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)

mbill = np.mat(bill) 
mtip = np.mat(tip)
m= np.shape(mbill)[1]
one = np.mat(np.ones(m))
X= np.hstack((one.T,mbill.T))

ypred = localWeightRegression(X,mtip,2)
SortIndex = X[:,1].argsort(0)
xsort = X[SortIndex][:,0]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(bill,tip, color='blue')
ax.plot(xsort[:,1],ypred[SortIndex], color = 'red', linewidth=1)
plt.xlabel('Total bill')
plt.ylabel('Tip')
