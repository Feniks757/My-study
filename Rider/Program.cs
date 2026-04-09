import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix


df_train = pd.read_csv("MNIST_train.csv")
df_test = pd.read_csv("MNIST_test.csv")

df_train = df_train.loc[df_train["label"].isin([0, 1])] # <-- Если в условии не 0 и 1, то поменять цифры
df_test = df_test.loc[df_test["label"].isin([0, 1])] # <-- Если в условии не 0 и 1, то поменять цифры

X_train = df_train.drop(["label"], axis=1)
X_test = df_test.drop(["label"], axis=1)
y_train = df_train["label"]
y_test = df_test["label"]

pca = PCA(svd_solver="full")
X_pca = pca.fit_transform(X_train)
min_value_ev = 0.9 # <-- Сюда долю объясненной дисперсии, если отличается

explained_variance = np.round(np.cumsum(pca.explained_variance_ratio_), 3)
n_components = 0
for i in range(len(explained_variance)):
    if (explained_variance[i] > min_value_ev):
        print("Количество главных компонент:", i)
        n_components = i
        break