#! /usr/bin/python3
#+ Autor:	Ran#
#+ Creado:	26/08/2020 19:05:27
#+ Editado:	26/08/2020 19:56:37

# importamos o dataset
from sklearn.datasets import load_iris
# importamos a función de división do dataset
from sklearn.model_selection import train_test_split
# importamos o algoritmo kNeightbors
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

SEPALO_L_MIN = 4.3
SEPALO_L_MAX = 7.9
SEPALO_A_MIN = 2.0
SEPALO_A_MAX = 4.4

PETALO_L_MIN = 1.0
PETALO_L_MAX = 6.9
PETALO_A_MIN = 0.1
PETALO_A_MAX = 2.5

# metemos o dataset nunha variable
iris = load_iris()

# random_state é a semente e poñémola para ter unha saída determinística
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

# instanciamos o algoritmo de entrenar e predicir
knn = KNeighborsClassifier(n_neighbors=1)

# construimos o modelo do set de entrenamento. colle os datos e as súas etiquetas.
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')

fiabilidade = knn.score(X_test, y_test)

print('-------------------')
print('Subespecies de Iris: {}, {}, {}.'.format(iris['target_names'][0],iris['target_names'][1],iris['target_names'][2]))
print('Fiabilidade do {}%\n'.format(round(fiabilidade*100,3)))


while True:
    sepalo_l = input('> Largura do sépalo: ')

    if sepalo_l.replace('.','').isdigit(): 
        sepalo_l = float(sepalo_l) 
        if sepalo_l >= SEPALO_L_MIN and sepalo_l <= SEPALO_L_MAX:
            break
    elif sepalo_l == '?':
        print('{} <= X <= {}\n'.format(SEPALO_L_MIN, SEPALO_L_MAX))

while True:
    sepalo_a = input('> Anchura do sépalo: ')
    if sepalo_a.replace('.','').isdigit():
        sepalo_a = float(sepalo_a)
        if sepalo_a >= SEPALO_A_MIN and float(sepalo_a) <= SEPALO_A_MAX:
            break
    elif sepalo_a == '?':
        print('{} <= X <= {}\n'.format(SEPALO_A_MIN, SEPALO_A_MAX))

while True:
    petalo_l = input('> Largura do pétalo: ')
    if petalo_l.replace('.','').isdigit():
        petalo_l = float(petalo_l)
        if petalo_l >= PETALO_L_MIN and float(petalo_l) <= PETALO_L_MAX:
            break
    elif petalo_l == '?':
        print('{} <= X <= {}\n'.format(PETALO_L_MIN, PETALO_L_MAX))

while True:
    petalo_a = input('> Anchura do pétalo: ')
    if petalo_a.replace('.','').isdigit():
        petalo_a = float(petalo_a)
        if petalo_a >= PETALO_A_MIN and float(petalo_a) <= PETALO_A_MAX:
            break
    elif petalo_a == '?':
        print('{} <= X <= {}\n'.format(PETALO_A_MIN, PETALO_A_MAX))

X_new = np.array([[sepalo_l, sepalo_a, petalo_l, petalo_a]])

print('\n> Subespecie de Iris: {}.'.format(iris['target_names'][knn.predict(X_new)][0]))
