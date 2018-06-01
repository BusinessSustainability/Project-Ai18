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
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier




scaler = StandardScaler()
names = ['Company_proficiency','Time_per_Unit','Innovation_Level','Quality','Stats','Sustainability_Index']
dataset= pd.read_csv("dataset4.csv",names=names)
X = dataset.drop('Sustainability_Index',axis=1)
y = dataset['Sustainability_Index']
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)
