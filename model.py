# This is the first Kaggle competition that I'm participating in for the sake of applyiny what I have learned from the Kaggle micro-courses.
# Hence, most of this code with not be mine since I am looking at highly-ranked Kaggle notebooks with in-depth, beginner-friendly explanations/walkthroughs.
# Credits:
#       Regularized Linear Models - https://www.kaggle.com/apapiu/regularized-linear-models/comments
#       https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f

# Importing packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score, train_test_split
import xgboost as xgb

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# print(train.head())
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
# matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({'price':train['SalePrice'], "log(price + 1)":np.log1p(train["SalePrice"])})
# prices.hist()
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

# model_ridge = Ridge() <-- I don't think we need this Ridge model...
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index=alphas)
# cv_ridge.plot(title= "Determining the best alpha parameter")
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.show()

print("Cross-validation RMSE of this Ridge Linear Model: {}".format(cv_ridge.min()))

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
print("Cross-validation RMSE of this Lasso Linear Model: {}".format(rmse_cv(model_lasso).mean()))
coef = pd.Series(model_lasso.coef_, index=X_train.columns)
print(f"Lasso picked {sum(coef != 0)} variables and eliminated the other {sum(coef == 0)} variables")
imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
# matplotlib.rcParams['figure.figsize'] = (14.0, 8.0)
# imp_coef.plot(kind = "barh")
# plt.title("Coefficients in the Lasso Model")
# plt.show()

matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
preds = pd.DataFrame({'preds':model_lasso.predict(X_train), 'true':y})
preds['residuals'] = preds['true'] - preds['preds']
preds.plot(x = 'preds', y = 'residuals', kind='scatter')
# plt.show()

# # Predicting on the test set
# predictions = np.expm1(model_lasso.predict(X_test))
# my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice':predictions})
# my_submission.to_csv('Submissions.csv', index=False)


enhanced_train = pd.concat([X_train, y])
print(enhanced_train)

dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

# *** Everything in this comment block was from the "Regularized Linear Models" Kaggle Notebook
# params = {'max_depth':2, 'eta':0.1}
# xgb_model = xgb.cv(params, dtrain, num_boost_round=500, early_stopping_rounds=10)
# xgb_model.loc[30:,['test-rmse-mean', 'train-rmse-mean']].plot()
# print(xgb_model[['test-rmse-mean']].idxmin())

params = {
    'max_depth':6,
    'min_child_weight': 1,
    'eta': .3,
    'subsample': 1,
    'colsample_bytree':1,
    'objective':'reg:linear',
    'eval_metric': 'rmse'
}
# print(train.shape[0])

# model_xgb = xgb.train(
#     params,
#     dtrain
# )
