# Importando Librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importing sets de datos
dataset = pd.read_csv('C:/Users/Dell/Documents/DataScience/Redes_Neuronales/Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Enconding categorical data
# Encoding the indepedent variable 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x_1 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:,1])

labelencoder_x_2 = LabelEncoder()
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])

onehotencoder = ColumnTransformer([("RowNumber", OneHotEncoder(), [1])], remainder = 'passthrough')
x = onehotencoder.fit_transform(x)
x = x[:,1:]

# spliting the dataset into  train set and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 0)

# Feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Parte 2 Creando red neuronal 

# Importando librerias y paquetes de Keras 


import keras
from keras.models import Sequential 
from keras.layers import Dense

# Iniciando red neuronal

clasificador = Sequential()

# Agregando capa input y primera capa oculta

clasificador.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Agregando segunda capa oculta

clasificador.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Agregando capa de salida

clasificador.add(Dense(units=6, kernel_initializer='uniform', activation='sigmoid'))

# compilando red neuronal / Decenso Gradiente Estocástica

clasificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Ajustando red neuronal en el set de entrenamiento

clasificador.fit(x_train, y_train, batch_size=10, epochs=100)

# Parte 3 Haciendo predicciones y evaluando el modelo

#Prediciendo sets de prueba 

y_pred = clasificador.predict(x_test)
y_pred = (y_pred > 0.5)

# Matriz de confusión

from sklearn.metrics import confusion_matrix
nc = confusion_matrix(y_test,y_pred)

# prediciendo nuevo cliente

