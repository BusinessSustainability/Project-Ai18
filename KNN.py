from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  sklearn.neighbors import KNeighborsClassifier

scaler = StandardScaler()
names = ['Company_proficiency','Time_per_Unit','Innovation_Level','Quality','Stats','Sustainability_Index']
dataset= pd.read_csv("dataset4.csv",names=names)
print(dataset.head())
print(dataset.describe().transpose())
print(dataset.shape)
X = dataset.drop('Sustainability_Index',axis=1)
Y = dataset['Sustainability_Index']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

#scaler.fit(X_train)
#StandardScaler(copy=True, with_mean=True, with_std=True)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

knn = KNeighborsClassifier()
knn.fit(X,Y)

predictions = knn.predict(X_test)
print(predictions)
from sklearn.metrics import classification_report,confusion_matrix
#print(confusion_matrix(y_test,predictions))
print(classification_report(Y_test,predictions))
