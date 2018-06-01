from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import linear_model
import statsmodels.api as sm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

scaler = StandardScaler()
names = ['Company_proficiency','Time_per_Unit','Innovation_Level','Quality','Stats','Sustainability_Index']
dataset= pd.read_csv("dataset4.csv",names=names)
print(dataset.head())
print(dataset.describe().transpose())
print(dataset.shape)
X = dataset.drop('Sustainability_Index',axis=1)
Y = dataset['Sustainability_Index']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)



model = XGBClassifier()
predictions = model.predict(X)
model.fit(X_train,Y_train)

accuracy=accuracy_score(Y_test,predictions)

print(model.summary())
