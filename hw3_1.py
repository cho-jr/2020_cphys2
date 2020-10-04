from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

boston = load_boston()
#print(boston.DESCR)


X = boston.data
y = boston.target


dfX = pd.DataFrame(X, columns = boston.feature_names)
dfy = pd.DataFrame(y, columns = ["MEDV"])
df = pd.concat([dfX, dfy], axis=1)
#print(df)
#print(df.describe())


reg = LinearRegression()
lasso_reg = Lasso(alpha = 5)
ridge_reg = Ridge(alpha = 5)

reg.fit(X, y)
lasso_reg.fit(X, y)
ridge_reg.fit(X, y)

#08 DIS, 직업센터 접근성
X_DIS = np.array(dfX["DIS"]).reshape(-1, 1)
y = np.array(dfy["MEDV"]).reshape(-1, 1)

reg.fit(X_DIS, y)
lasso_reg.fit(X_DIS, y)
ridge_reg.fit(X_DIS, y)

"""
#Linear Regression

plt.title("Linear Regression")
plt.scatter(X_DIS, y, s=5, label= "DIS")
plt.plot(X_DIS, reg.predict(X_DIS),label = 'reg_fit') 
plt.xlabel("DIS")
plt.ylabel("price")
plt.legend()
plt.show()

#Lasso Regression

plt.title("Lasso Regression")
plt.scatter(X_DIS, y, s=5, label= "DIS")
plt.plot(X_DIS, lasso_reg.predict(X_DIS),label = 'lasso_fit') 
plt.xlabel("DIS")
plt.ylabel("price")
plt.legend()
plt.show()


#Ridge Regression

plt.title("Ridge Regression")
plt.scatter(X_DIS, y, s=5, label= "DIS")
plt.plot(X_DIS, ridge_reg.predict(X_DIS),label = 'ridge_fit') 
plt.xlabel("DIS")
plt.ylabel("price")
plt.legend()
plt.show()
"""

#슴

plt.title("Boston House Price")
plt.scatter(X_DIS, y, s=5, label= "DIS")

plt.plot(X_DIS, reg.predict(X_DIS),color='cyan', linestyle='dashed' ,label = 'reg_fit') 
plt.plot(X_DIS, lasso_reg.predict(X_DIS),color='magenta',linestyle='dotted',label = 'lasso_fit') 
plt.plot(X_DIS, ridge_reg.predict(X_DIS),color='red', linestyle=':', label = 'ridge_fit') 

plt.xlabel("DIS")
plt.ylabel("price")
plt.legend()
plt.show()
