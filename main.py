import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

veri = pd.read_csv("EVDS.csv")
veri.head()

x = veri["Tarih"].values
y = veri["TP DK USD A YTL"].values

X = x.reshape(23, 1)
Y = y.reshape(23, 1)

##lineer reg.
X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

tahminlineer = LinearRegression()
tahminlineer.fit(X_train, Y_train)
y_pred= tahminlineer.predict(X_test)

plt.scatter(X_train, Y_train, color='blue')
plt.plot(X_train, tahminlineer.predict(X_train), color="black")

##polinom reg.
tahminpol = PolynomialFeatures(degree=3)

Xyeni = tahminpol.fit_transform(X_train)
tahminpol.fit(Xyeni,Y_train)

danan = LinearRegression()
danan.fit(Xyeni,Y_train)

X_grid = X.reshape(23,1)

plt.scatter(X_train,Y_train, color="blue")
plt.plot(X_grid, danan.predict(tahminpol.fit_transform(X_grid)),
         color="red")
plt.show()
