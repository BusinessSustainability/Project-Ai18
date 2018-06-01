from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.svm import SVR
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.linear_model import LinearRegression




scaler = StandardScaler()
names = ['Company_proficiency','Time_per_Unit','Innovation_Level','Quality','Stats','Sustainability_Index']
dataset= pd.read_csv("dataset4.csv",names=names)

#plot(sns.pairplot(dataset, x_vars=['Company_proficiency','Time_per_Unit','Innovation_Level','Quality','Stats'], y_vars='Sustainability_Index', size=7, aspect=0.7))
lm1 = smf.ols(formula='Sustainability_Index ~ Company_proficiency + Time_per_Unit + Innovation_Level + Quality + Stats', data=dataset).fit()
print(lm1.params)
X = dataset.drop('Sustainability_Index',axis=1)
y = dataset['Sustainability_Index']
lm2 = LinearRegression()
lm2.fit(X, y)
print(lm2.intercept_)
print(lm2.coef_)
print(lm1.summary())
