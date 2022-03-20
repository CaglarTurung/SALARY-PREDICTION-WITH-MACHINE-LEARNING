import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import random
from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR

# for making output full :
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

df_=pd.read_pickle("E:\CAGLAR\HITTERS_DATA_PREP.pkl")
df=df_.copy()
df.head()
check_df(df)

# MODELING

## Linear Regression
y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)

reg_model = LinearRegression().fit(X, y)
y_pred = reg_model.predict(X)
np.sqrt(mean_squared_error(y, y_pred)
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))
# 305.041

## XG BOOST
y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)
xg_model=XGBRegressor().fit(X,y)
np.mean(np.sqrt(-cross_val_score(xg_model, X, y, cv=10, scoring="neg_mean_squared_error")))

# 292.64

## GBM
y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)

gbm_model=GradientBoostingRegressor().fit(X,y)
np.mean(np.sqrt(-cross_val_score(gbm_model, X, y, cv=10, scoring="neg_mean_squared_error")))

# 278.31

## Light GBM
y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)

light_gbm_model = LGBMRegressor().fit(X,y)
y_pred = light_gbm_model.predict(X)

np.mean(np.sqrt(-cross_val_score(light_gbm_model, X, y, cv=10, scoring="neg_mean_squared_error")))

# 281.63

## Random Forest
y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)

rf_model = RandomForestRegressor().fit(X,y)

np.mean(np.sqrt(-cross_val_score(rf_model, X, y, cv=10, scoring="neg_mean_squared_error")))

# 263.14 # The best rmse score


# Hyperparameter Optimization

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

rf_final = rf_model.set_params(**rf_best_grid.best_params_,
                               random_state=17).fit(X, y)

np.mean(np.sqrt(-cross_val_score(rf_final, X, y, cv=10, scoring="neg_mean_squared_error")))

# 260.57 # Rmse score after hyperparameter optimization.

# Let's predict a random hitter's salary

random_user = X.sample(1, random_state=42)

rf_final.predict(random_user)  # 746.57

################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X) 