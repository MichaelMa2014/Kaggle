import sklearn.tree

from data import DataHandler

d = DataHandler()
print(d.train)
print(d.target)
print(d.test)

classifier = sklearn.tree.DecisionTreeClassifier()
classifier.fit(d.train, d.target)
prob = classifier.predict_proba(d.test)
d.save_submission(prob)

