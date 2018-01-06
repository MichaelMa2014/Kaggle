import os
import sklearn.tree
import sklearn.metrics
import sklearn.ensemble

from util import OUTPUT_PATH
from data import DataHandler

d = DataHandler()
print(d.train)
print(d.target)
print(d.test)

classifier = sklearn.tree.DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=39)
classifier.fit(d.train, d.target)
prob = classifier.predict_proba(d.test)
d.save_submission(prob, "decision_tree")
print("decision_tree: Train set acc:", classifier.score(d.train, d.target))
print("decision_tree: Train set loss:", sklearn.metrics.log_loss(d.target, classifier.predict_proba(d.train)))
sklearn.tree.export_graphviz(
        classifier,
        out_file=os.path.join(OUTPUT_PATH, "tree.dot"),
        label="none",
        impurity=False,
        filled=True)

ada_classifier = sklearn.ensemble.AdaBoostClassifier(classifier)
ada_classifier.fit(d.train, d.target)
prob = ada_classifier.predict_proba(d.test)
d.save_submission(prob, "decision_tree_ada")
print("decision_tree_ada: Train set acc:", ada_classifier.score(d.train, d.target))
print("decision_tree_ada: Train set loss:", sklearn.metrics.log_loss(d.target, ada_classifier.predict_proba(d.train)))

