import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, x_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

def checknull(data_f, var):
    print(var, data_f[var].isnull().sum())
  #  print(data_f[var].value_counts())
    sums = data_f[var].isnull().sum()+data_f[var].value_counts().sum()
    print(int(sums), train.shape[0])
    print("-"*50)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

## Looking into the Alley variable
## convert the NaN to NAs for earier access

train.Alley[train.Alley.isnull()] = "NA"
fig1 = plt.figure()

plt.hist(train.SalePrice[train.Alley=='NA'], normed=True, 
    color='green', alpha=0.5, bins=50, label='NA')
plt.hist(train.SalePrice[train.Alley=='Grvl'], normed=True, 
    color='yellow', alpha=0.5, bins=50, label='Grvl')
plt.hist(train.SalePrice[train.Alley=='Pave'], normed=True, 
    color='blue', alpha=0.5, bins=50, label='Pave')
plt.legend(loc='best')
plt.xlabel("Sale Price")

condition = train.Alley
fig2 = plt.figure()
plt.scatter(train.SalePrice[condition=='Grvl'], 
            train.LotArea[condition=='Grvl'], color='yellow', 
            alpha=0.7, label='Grvl')
plt.scatter(train.SalePrice[condition=='Pave'], 
            train.LotArea[condition=='Pave'], color='blue', 
            alpha=0.7, label='Pave')
plt.scatter(train.SalePrice[condition=='NA'], 
            train.LotArea[condition=='NA'], color='green',
            alpha=0.3, label='NA')
plt.legend(loc='best')
plt.xlabel("Lot Area")
plt.ylabel("Sale Price")

base_vars = [s for s in train.columns if 'Bsmt' in s]
for base in base_vars:
    checknull(train, base)