#Simple but relatively slow since it has to iterate over every data-point.

import random
from scipy.spatial import distance
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

def euc(a,b):
  return(distance.euclidean(a,b))

class Classifier():

  def fit(self,train,train_labels):
    self.train = train
    self.train_labels = train_labels

  def closest(self,x):
    best_dist=euc(x,self.train[0])
    best_index = 0 
    for i in range(len(self.train)):
      dist=euc(x,self.train[i])
      if(dist<best_dist):
        best_dist=dist
        best_index=i
    return(self.train_labels[best_index])

  def predict(self,test):
    predictions = []
    for i in test:
      label = self.closest(i)
      predictions.append(label)
    return(predictions)
    
iris = load_iris()
features = iris.data
labels = iris.target
train,test,train_labels,test_labels = train_test_split(features,labels,test_size=0.5)
classifier = Classifier()
classifier.fit(train,train_labels)
pred = classifier.predict(test)
print(accuracy_score(test_labels,pred))
