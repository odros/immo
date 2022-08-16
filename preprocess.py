"""
This script preprocesses
"""
# Import modules
import os, pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Read the data
os.chdir("/Users/aleph/Desktop/MDS/semestres/2/ml/final/data/")
data = pd.read_csv("curated_data.csv")

# Drop unique identifier
data = data.drop(['obid'], axis = 1)

# Create one-hot encoded 'adat'
one_hot_adat = pd.get_dummies(data['adat'])

# Drop unencoded column
data = data.drop('adat', axis = 1)

# Scale data before pasting 'adat' back, we only fit to the train data!
names = data.columns
scaler = MinMaxScaler()
scaler.fit(data)
data = pd.DataFrame(scaler.transform(data), columns = names)

# Paste 'adat' back
data = data.join(one_hot_adat)

## Prepare the predictor vector 'X' and labels 'y', make sure to leave out the 'obid' variable out
X = data.drop(['hits'], axis = 1)
y = data['hits']

## Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)
