# This is the first Kaggle competition that I'm participating in for the sake of applyiny what I have learned from the Kaggle micro-courses.
# Hence, most of this code with not be mine since I am looking at highly-ranked Kaggle notebooks with in-depth, beginner-friendly explanations/walkthroughs.
# Credits:
#       Comprehensive data exploration with Python by Pedro Marcelino - https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

# Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

df_train = pd.read_csv('train.csv')
print(df_train['SalePrice'].describe())
sns.distplot(df_train['SalePrice'])
plt.show()