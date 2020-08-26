#! /usr/bin/python3
#+ Autor:	Ran#
#+ Creado:	25/08/2020 18:38:34
#+ Editado:	26/08/2020 18:35:34

import numpy as np
# importamos o dataset
from sklearn.datasets import load_iris
# importamos a función de división do dataset
from sklearn.model_selection import train_test_split
# importamos o algoritmo kNeightbors
from sklearn.neighbors import KNeighborsClassifier


# metemos o dataset nunha variable
iris = load_iris()

# random_state é a semente e poñémola para ter unha saída determinística
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

# instanciamos o algoritmo de entrenar e predicir
knn = KNeighborsClassifier(n_neighbors=1)

# construimos o modelo do set de entrenamento. colle os datos e as súas etiquetas.
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')

## agora podemos facer prediccións en novos datos dos cales podemos non saber as etiquetas correspondentes

# un exemplo
X_new = np.array([[5, 2.9, 1, 0.2]])
# X_new.shape (1, 4)

# facemos a predición
prediction = knn.predict(X_new)
# de que tipo será o noso exemplo
print(prediction)
# que nome recibe o número de tipo
print(iris['target_names'][prediction])

## evaluar o modelo

# facemos a predicción dos datos de testeo
y_pred = knn.predict(X_test)

# facemos a media dos acertos (porcentaxe de exactitude do modelo)
print(np.mean(y_pred == y_test))
# usando esta forma non fai falla facer a predicción
print(knn.score(X_test, y_test))


