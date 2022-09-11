import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")
species = list(set(df["Species"]))
df["Species_indexed"] = pd.factorize(df["Species"])[0]
df = df.drop(["Id", "Species"], axis=1)
df_minmax = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df[df.columns[:-1]]), columns = df.columns[:-1])
X_train, X_test, y_train, y_test = train_test_split(df_minmax, df["Species_indexed"], test_size=0.33, random_state=420)

