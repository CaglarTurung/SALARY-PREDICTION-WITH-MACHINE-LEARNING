import pandas as pd
from sklearn.preprocessing import RobustScaler
from helpers.eda import *
from helpers.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

df_=pd.read_csv("E:\CAGLAR\datasets\hitters.csv")
df=df_.copy()
df.head()
# DATA PREPROCESSING
check_df(df)
# There are missing values in the salary variable.
df.describe().T # Descriptive Statistics
 # There seems to be outlier values.

## Missing Values

df.isnull().sum()

# Missing values are in the dependent variable, we may prefer to delete the missing values.
df.dropna(inplace=True)
df.isnull().sum()

## Capital Letters
df.columns = [col.upper() for col in df.columns]
df.columns

## The variables were divided into categorical and numerical.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols.remove('SALARY')

for col in cat_cols:
    cat_summary(df, col)

for col in cat_cols:
        target_summary_with_cat(df, 'SALARY', col)

## Outlier Values

for col in num_cols:
    print(col, check_outlier(df, col)) # There are no outlier values.

## The relationship of numerical variables with the target variable.

target_correlation_matrix(df, target="SALARY",  corr_th=0.50, plot=True)

# ['CATBAT', 'CHITS', 'CHMRUN', 'CRUNS', 'CRBI', 'SALARY'] # The variable with the highest correlation is "CRBI".

##### FEATURE ENGINEERING #####

## YEARS
df.loc[(df["YEARS"] <= 1 ), "NEW_EXPERIENCE"] = "ROOKIE"
df.loc[(df["YEARS"] > 1 ) & (df["YEARS"] <= 5), "NEW_EXPERIENCE"] = "FRESH"
df.loc[(df["YEARS"] > 5 ) & (df["YEARS"] <= 10), "NEW_EXPERIENCE"] = "STARTER"
df.loc[(df["YEARS"] > 10 ) & (df["YEARS"] <= 15), "NEW_EXPERIENCE"] = "AVERAGE"
df.loc[(df["YEARS"] > 15 ) & (df["YEARS"] <= 20), "NEW_EXPERIENCE"] = "EXPERIENCED"
df.loc[(df["YEARS"] > 20 ), "NEW_EXPERIENCE"] = "VETERAN"

df.head()

## HITS
df.sort_values("HITS", ascending=False).head()
df["NEW_HITS_CALSS"] = pd.qcut(df["HITS"], 4, labels=["D","C","B","A"])
df.head()

## ['CATBAT', 'CHITS', 'CHMRUN', 'CRUNS', 'CRBI', 'SALARY']
df['NEW_HIT_RATIO'] = df['HITS'] / df['ATBAT'] # Average number of successful hits
df['NEW_RUN_RATIO'] = df['HMRUN'] / df['RUNS']
df['NEW_RUN_RATIO'].isnull().sum()

df['NEW_RUN_RATIO'].fillna(0, inplace=True)
df['NEW_RUN_RATIO'].isnull().sum()
df['NEW_CHIT_RATIO'] = df['CHITS'] / df['CATBAT']
df['NEW_CRUN_RATIO'] = df['CHMRUN'] / df['CRUNS']

df['NEW_AVG_CATBAT'] = df['CATBAT'] / df['YEARS']
df['NEW_AVG_HITS'] = df['CHITS'] / df['YEARS']
df['NEW_AVG_HMRUN'] = df['CHMRUN'] / df['YEARS']
df['NEW_AVG_RUNS'] = df['CRUNS'] / df['YEARS']
df['NEW_AVG_RBI'] = df['CRBI'] / df['YEARS']
df['NEW_AVG_WALKS'] = df['CWALKS'] / df['YEARS']

# SCALING & ONE HOT ENCODING
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

# ONE-HOT ENCODING
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)
df.head()

# ROBUST SCALER
rs = RobustScaler()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols.remove('SALARY')
df[num_cols] = rs.fit_transform(df[num_cols])

df.head()

df.to_pickle("./HITTERS_DATA_PREP.pkl")
