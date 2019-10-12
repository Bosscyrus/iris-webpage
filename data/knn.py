from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
iris = load_iris()

x= iris.data
y= iris.target

x,y = shuffle(x,y,random_state = 0)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x,y)

#import pickle
#filename = 'iris_knn.sav'
#pickle.dump(knn, open(filename, 'wb'))
