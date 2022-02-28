import pandas as pd

import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    r'C:\Users\49176\OneDrive\Documents\AIprojects\ML_bauen\data\auto-mgp.csv', sep=';')

x = df.loc[:, df.columns != 'mpg']
y = df["mpg"]
print(df.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

reg = LinearRegression().fit(x_train, y_train)


filename = 'finalized_model.sav'
pickle.dump(reg, open(filename, 'wb'))
