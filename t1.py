1. FIND S
import pandas as pd
import numpy as np
 
data = pd.read_csv("FindS.csv")
X = data.iloc[:,:].values
 
hypo = ['0']*int(len(X[0])-1)
 
for i,h in enumerate(X):
 if h[-1]=="yes":
   for j in range(len(hypo)):
     if hypo[j]=="0":
       hypo[j]=h[j]
     elif h[j]!=hypo[j]:
       hypo[j]="?"
 
print("The specific hypothesis is: ",hypo)

2.CANDIDATE
import pandas as pd
import numpy as np
 
data = pd.read_csv("Candidate_Elimination.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
 
s = ['0' for i in range(len(X[0]))]
g = [['?' for i in range(len(X[0]))] for i in range(len(X[0]))]
 
for i,h in enumerate(X):
 if y[i]=="yes":
   s = h.copy()
   break
 
for i,h in enumerate(X):
 if y[i]=="yes":
   for j in range(len(s)):
     if h[j]!=s[j]:
       s[j]="?"
       g[j][j]="?"
 elif y[i]=="no":
   for j in range(len(s)):
     if h[j]!=s[j]:
       g[j][j] = s[j]
     else:
       g[j][j] = "?"
 
temp = ["?" for x in range(len(X[0]))]
general_h = []
for i in g:
 if i!=temp:
   general_h.append(i)
 
specific_h = s.copy()
 
print("The most specific hypothesis is: ",specific_h)
print("The most general hypothesis is: ",general_h)

3.KNN
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
 
 
data =load_iris()
 
iris=load_iris()
x=iris.data
y=iris.target
 
x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.4,random_state=5)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=5)
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Acuraccy of KNN\t\t\t\t",accuracy_score(y_pred,y_test))
 
pd.DataFrame({'actual':y_test,'predction':y_pred,'correct classification':(y_test==y_pred)})

4.NAIVE BAIYES
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
 
data = pd.read_csv("naive_dataset.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
 
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
X[:,2] = le.fit_transform(X[:,2])
X[:,3] = le.fit_transform(X[:,3])
X[:,4] = le.fit_transform(X[:,4])
y = le.fit_transform(y)
 
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
 
model = GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("The accuracy score of Naive Bayes Classifier is: ", format(accuracy_score(y_test,y_pred),"0.3f"))

5.LWR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("tips.csv")
bill = data.total_bill
tip = data.tip

mBill = np.mat(bill)
mTip = np.mat(tip)
m = mBill.shape[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T, mBill.T))

def kernel(point,xmat,k):
  m,n = xmat.shape
  weights = np.mat(np.eye(m))
  for j in range(m):
    diff = point - xmat[j]
    weights[j,j] = np.exp(diff*diff.T/(-2*k**2))
  return weights

def Beta(x_value,x,y,k):
  weight = kernel(x_value,x,k)
  W = (X.T * (weight * X)).I*(X.T *(weight * y.T))
  return W

def localWeightRegression(x,y,k):
  m,n = x.shape
  ypred = np.zeros(m)
  for i in range(m):
    ypred[i] = x[i] * Beta(x[i],x,y,k)
  return ypred

ypred = localWeightRegression(X,mTip,2)
SortIndex = X[:,1].argsort(0)
xsort = X[SortIndex][:,0]

plt.figure()
plt.scatter(bill,tip,color='blue')
plt.plot(xsort[:,1],ypred[SortIndex],color='red',linewidth = 2)
