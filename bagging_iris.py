from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble import BaggingClassifier
dt=DecisionTreeClassifier(random_state=42)
lr=LogisticRegression()
base_estimator=dt,lr
#_____________________________________________________
bagging_model=BaggingClassifier(
    estimator=base_estimator,
    n_estimators=15,# number of trees
    random_state=20
)

df=datasets.load_iris()
x=df["data"]
y=df["target"]
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=10000)
bagging_model.fit(x,y)


