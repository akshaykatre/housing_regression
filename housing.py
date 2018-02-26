import pandas as pd 
import numpy as np 
import seaborn as sns 
from matplotlib import pyplot as plt

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

def categ_dist_mis(train_data, cats, test_data=[]):
    for cat in cats:
        print(cat)
        print("_"*40)
        print("In train: ")
        print(train_data[cat].value_counts(dropna = False))
        print('_'*40)
        if test_data != []:
            print("In test: ")
            print(test_data[cat].value_counts(dropna = False))
            print("_"*40)
        print("_"*40)


def traintest_hist(train_data, feat, nbins, test_data = []):
    fig, axes = plt.subplots(1, 2)

    train_data[feat].hist(bins = nbins, ax=axes[0])
   
    print("{}: {} missing, {}%".format('Train', train_data[feat].isnull().sum(),
                                  round(train_data[feat].isnull().sum()/train_data.shape[0] * 100, 3)))

    if test_data != []:
        df_test[feat].hist(bins = nbins, ax=axes[1])
        print("{}: {} missing, {}%".format('Test', df_test[feat].isnull().sum(),
                                  round(df_test[feat].isnull().sum()/df_test.shape[0] * 100, 3)))


