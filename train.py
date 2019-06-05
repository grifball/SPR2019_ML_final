import ember
import lightgbm as lgb
import sys

data_dir = "./data/ember/"
# this function mmaps the data into memory
X_train, y_train = ember.read_vectorized_features(data_dir, subset="train")
# seems that the data gets sorted by the label (so the end is full of label 1, start is label 0, dunno where label -1 is, but that's my job)
# grab only 2000 points
# filtering through all the points takes forever, you have to seek to get data quickly
X_subset_train = X_train[0:1000] + X_train[900000-1000:900000]
y_subset_train = y_train[0:1000] + y_train[900000-1000:900000]
# setup the dataset from lgbm (I just c&p'd this from the ember code)
lgbm_dataset = lgb.Dataset(X_subset_train, y_subset_train)
# run nthe lightgbm model
lgbm_model = lgb.train({"application": "binary"}, lgbm_dataset)
# print out some classifications (should all be 0 cause it's the start of the data)
print(lgbm_model.predict(X_train[1000:2000]))
