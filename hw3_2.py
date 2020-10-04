from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dia = load_diabetes()
print(dia.DESCR)

X = dia.data
y = dia.target

#print(dia.feature_names)
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
#s6 = 혈당

#Regression
reg = LinearRegression()
lasso_reg = Lasso(alpha = 5)
ridge_reg = Ridge(alpha = 5)

dfX = pd.DataFrame(X, columns = dia.feature_names)
dfy = pd.DataFrame(y, columns = ["progression"])
df = pd.concat([dfX, dfy], axis=1)


X_glu = np.array(dfX['s6']).reshape(-1, 1)

reg.fit(X_glu, y)
lasso_reg.fit(X_glu, y)
ridge_reg.fit(X_glu, y)


#슴

plt.title("Diabetes")
plt.scatter(X_glu, y, s=5, label= "Blood Sugar Level")

plt.plot(X_glu, reg.predict(X_glu),color='cyan', linestyle='dashed' ,label = 'reg_fit') 
plt.plot(X_glu, lasso_reg.predict(X_glu),color='magenta',linestyle='dotted',label = 'lasso_fit') 
plt.plot(X_glu, ridge_reg.predict(X_glu),color='red', linestyle=':', label = 'ridge_fit') 

plt.xlabel("Blood Sugar Level")
plt.ylabel("progression")
plt.legend()
plt.show()
