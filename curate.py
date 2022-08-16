### Recode and curate
import pandas as pd
import numpy as np
import os

# Read the data
os.chdir("/Users/aleph/Desktop/MDS/semestres/2/ml/data/")
data = pd.read_csv("raw_data.csv", sep = ";")

## Recode all missing data to NaN
missing = [-11, -10, -9, -8, -7, -6, -5]
data = data.replace(missing, np.nan)

## Impute data for variables with low NaN count

# What is the NaN count for every predictor?
nans = data.isna().sum()

# 'zimmeranzahl' has too few to drop, impute the model
data["zimmeranzahl"].isna().sum()
zimmeranzahl_mode = data["zimmeranzahl"].mode()[0]
data["zimmeranzahl"].fillna(zimmeranzahl_mode, inplace = True)

# 'nebenkosten' has too few to drop, impute the mean
data["nebenkosten"].isna().sum()
nebenkosten_mean = data["nebenkosten"].mean()
data["nebenkosten"].fillna(nebenkosten_mean, inplace = True)

## Keep only columns with zero NaNs

# Subset dataset to keep only variables without NaNs
filter = data.isna().sum() == 0
data = data.loc[:, filter]

## Some feature engineering and scaling

# Remove all click variables, since they are metainformation
data = data.drop(['click_schnellkontakte', 'click_weitersagen', 'click_url'], axis = 1)

# Drop more metainformation and geoinformation for now, keeping 'obid' to be able to index specific cases
data = data.drop(['gid2019', 'kid2019', 'immobilientyp', 'lieferung'], axis = 1)

## Write the data to a CSV
os.chdir("/Users/aleph/Desktop/MDS/semestres/2/ml/final/data/")
data.to_csv("curated_data.csv", index = False)
