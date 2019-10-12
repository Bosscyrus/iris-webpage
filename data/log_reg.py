from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression

log = LogisticRegression
iris = load_iris()

x= iris.data
y= iris.target

log.fit(x,y)

#import pickle
#filename = 'iris_log.sav'
#pickle.dump(log, open(filename, 'wb'))