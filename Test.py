from sklearn.ensemble import IsolationForest
X = [[-1.1], [0.3], [0.5], [1]]
clf = IsolationForest(random_state=0).fit(X)
print(clf.predict([[0.1], [0], [90]]))
print(clf.score_samples([[0.1], [0], [90]]))