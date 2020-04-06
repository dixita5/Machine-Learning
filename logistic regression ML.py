import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
dataset= load_iris()
X=dataset.data
y=dataset.target

plt.scatter(X[y==0,0],X[y==0,1],c="r",label="Setosa")
plt.scatter(X[y==1,0],X[y==1,1],c="g",label="Versicolour")
plt.scatter(X[y==2,0],X[y==2,1],c="b",label="Verginica")
plt.xlabel('sepal Length')
plt.ylabel('sepal width')
plt.title('analysis on iris datasets')
plt.legend()
plt.show()
%matplotlib qt5


plt.scatter(X[y==0,2],X[y==0,3],c="r",label="Setosa")
plt.scatter(X[y==1,2],X[y==1,3],c="g",label="Versicolour")
plt.scatter(X[y==2,2],X[y==2,3],c="b",label="Verginica")
plt.xlabel('Petal Length')
plt.ylabel('Petal width')
plt.title('analysis on iris datasets')
plt.legend()
plt.show()

test=pd.DataFrame(X)
pd.plotting.scatter_matrix(test)
test.columns=['sepal length','sepal width','petal length','petal width']


from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X,y)
log_reg.score(X,y)

x=np.arange(-10,10,0.01)
sig_y= 1/(1+np.power(np.e, -x))
sig_yy= np.power(np.e,-x)/(1+np.power(np.e, -x))
plt.plot(x,sig_y)
plt.show()

plt.plot(x,sig_yy)
plt.show()



lop= 4*x + 7
plt.plot(x,lop)
plt.show()

sig_y_lop= 1/(1+np.power(np.e, lop))
sig_yy_lop= np.power(np.e,lop)/(1+np.power(np.e, lop))
plt.plot(x,sig_y_lop)
plt.show()
plt.plot(x,sig_yy_lop)
plt.show

y_pred=log_reg.predict(X)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y,y_pred)
log_reg.score(X,y)
