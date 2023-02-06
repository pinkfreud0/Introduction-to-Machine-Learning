import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('CarPrice_Assignment.csv')
data.head()


# sns.set_style("whitegrid")
# plt.figure(figsize=(15, 10))
# sns.distplot(data.price)
# plt.show()