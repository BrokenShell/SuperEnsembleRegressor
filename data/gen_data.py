import os
from string import ascii_uppercase

import numpy as np
from pandas import DataFrame
from sklearn import datasets


n_features = 7
n_samples = 5000
n_outliers = 1000
target = "Target"
features = list(ascii_uppercase)[:n_features]

X, y, coef = datasets.make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=1,
    noise=10,
    coef=True,
    random_state=42,
)
np.random.seed(0)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

df = DataFrame(data=X, columns=features)
df[target] = y
df.head()

df.to_csv(os.path.join("dataset3.csv"), index=False)
