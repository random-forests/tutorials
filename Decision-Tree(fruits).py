from sklearn import tree
features = [[140,0],[130,0],[150,1],[170,1]] 
labels = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
print(clf.predict([[120,0]])) #Output 0
print(clf.predict([[180,1]])) #Output 1