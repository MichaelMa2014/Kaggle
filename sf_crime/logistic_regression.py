import sklearn.linear_model

from util import OUTPUT_PATH
from data import DataHandler

d = DataHandler()
print(d.train)
print(d.target)
print(d.test)

classifier = sklearn.linear_model.LogisticRegression(
        penalty="l2",
        solver="newton-cg",
        multi_class="multinomial",
        n_jobs=-1)
classifier.fit(d.train, d.target)
prob = classifier.predict_proba(d.test)
d.save_submission(prob, "logistic_regression_newton-cg_multinomial")
print("logistic_regression: Train set acc:", classifier.score(d.train, d.target))
print("logistic_regression: Train set loss:", sklearn.metrics.log_loss(d.target, classifier.predict_proba(d.train)))
