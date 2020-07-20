# This is the first Kaggle competition that I'm participating in for the sake of applyiny what I have learned from the Kaggle micro-courses.
# Hence, most of this code with not be mine since I am looking at highly-ranked Kaggle notebooks with in-depth, beginner-friendly explanations/walkthroughs.
# Credits:
#       Regularized Linear Models - https://www.kaggle.com/apapiu/regularized-linear-models/comments

# Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# print(train.head())
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({'price':train['SalePrice'], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()
#plt.show()

train["SalePrice"] = np.log1p(train["SalePrice"])

numerical_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numerical_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75].index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring='neg_mean_squared_error', cv=5))
    return rmse
