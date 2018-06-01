from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

scaler = StandardScaler()
names = ['Company_proficiency','Time_per_Unit','Innovation_Level','Quality','Stats','Sustainability_Index']
dataset= pd.read_csv("dataset4.csv",names=names)
print(dataset.head())
print(dataset.describe().transpose())
print(dataset.shape)
X = dataset.drop('Activity',axis=1)
Y = dataset['Sustainability_Index']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

loo = LeaveOneOut()
loo.get_n_splits(X)



KFold(n_splits=20, random_state=None, shuffle=False)


knn = KNeighborsClassifier()
knn.fit(X,Y)

predictions = knn.predict(X_test)
print(predictions)
from sklearn.metrics import classification_report,confusion_matrix
#print(confusion_matrix(y_test,predictions))
print(classification_report(Y_test,predictions))
