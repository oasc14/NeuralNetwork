# Pre Procesamiento de Datos

# Importando Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing set de datos
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Arreglando Datos Nulos
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Codificando Datos Categoricos
# Codificando Variable Independiente
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Codificando Variable Dependiente
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)