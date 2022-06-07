import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv('data/auction_stats.csv')
X = df.drop(columns=['player_pkey', 'Player', 'Team', 'Year', 'Amount'])
y = df['Amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

result = pd.DataFrame()

models = [LinearRegression(), BayesianRidge(), SVR(C=1.0, epsilon=0.2),
          XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8),
         RandomForestRegressor()]

for model in models:
    model.fit(X_train, y_train)
    a = model.score(X_train, y_train)
    b = model.score(X_test, y_test)
    result = result.append({'name': str(model), 'Train': a, 'Test': b}, ignore_index=True)

print(result)