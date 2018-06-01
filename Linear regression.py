from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import linear_model
import statsmodels.api as sm

scaler = StandardScaler()
names = ['Company_proficiency','Time_per_Unit','Innovation_Level','Quality','Stats','Sustainability_Index']
dataset= pd.read_csv("dataset4.csv",names=names)
print(dataset.head())
print(dataset.describe().transpose())
print(dataset.shape)
X = dataset.drop('Sustainability_Index',axis=1)
Y = dataset['Sustainability_Index']
lm = linear_model.LinearRegression()
model = lm.fit(X,Y)

predictions = lm.predict(X)
print(predictions)

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print(model.summary())
