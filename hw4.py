## hw4 - Iris

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()
"""
key = iris.keys()
print(key)
#['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename']
print(iris.target)
print(iris.target_names) 

#['setosa'
# 'versicolor'
#'virginica']

fn = iris.feature_names
print(fn)
#['sepal length (cm)',
# 'sepal width (cm)',
# 'petal length (cm)',
# 'petal width (cm)']
print(iris.DESCR)
"""
X = [[i] for i in iris.data[:, 2]] # 꽃잎 길이=2
y = [1 if i==0 else 0 for i in iris.target] #i==0 --> 세토사

log_reg = LogisticRegression(solver = 'lbfgs')
log_reg.fit(X, y)
score = log_reg.score(X, y)
print(score)

x = [[i] for i in np.linspace(1, 6, 100)]
plt.title("Fig.1 logistic regression, petal length, setosa")
plt.xlabel('Petal Length')
plt.ylabel('1=Setosa')
plt.plot(X, y, 'bo', label = '1: setosa \n0: else')
plt.plot(x, log_reg.predict(x), 'g-', label = 'log regression')
plt.legend()
plt.show()


###########################
#classifier
#Support Vector Machine
from sklearn.svm import SVC

svm = SVC(kernel = 'rbf', gamma = 'auto')
svm.fit(X, y)
svm_score1 = svm.score(X, y)
print(svm_score1)
plt.title("Fig.2 Support Vector Classification fitting")
plt.scatter(X, y, c = svm.predict(X))
plt.xlabel('Petal Length')
plt.ylabel('1=Setosa')
#plt.legend()
plt.show()

X = iris["data"]
y = iris["target"]	# setosa=0, versicolor=1, virginica=2

svm = SVC(kernel = 'rbf', gamma = 'auto')
svm.fit(X, y)
svm_score2 = svm.score(X, y)
print(svm_score2)

plt.title("Fig.3 Suport Vector Machine Prediction")
label = ['setosa', 'versicolor', 'virginica']
color_list = list(set(svm.predict(X)))

plt.scatter(X[:, 1], X[:, 3], c = svm.predict(X))
plt.xlabel("Sepal width")
plt.ylabel("petal width")
plt.legend({'setosa', 'versicolor', 'virginica'})
plt.show()

"""
짧은 보고서
LogisticRegression을 이용해서 Iris 데이터를 분류했다.
Fig.1 에서 꽃잎 길이를 이용해 세토사를 다른 데이터들과 분리시켰다. 
Fig.2 에서 같은 분류 작업에 SVC를 이용했다. 
Fig.3 에서 x 축을 꽃받침 너비, y 축을 꽃잎 너비로 하고 scatter 하면
세토사를 다른 종들과 명확히 분류할 수 있다.  
"""
