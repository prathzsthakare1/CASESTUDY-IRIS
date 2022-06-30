import pandas as pd

df = pd.read_csv("iris.data")

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

rfc = RandomForestClassifier(random_state=1)
logr = LogisticRegression(random_state=0)
gbc = GradientBoostingClassifier(n_estimators=10)
dtc = DecisionTreeClassifier(random_state=0)
svm = svm.SVC()
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)
nb = MultinomialNB()

x=df.drop('Species',axis=1)
y=df['Species']
x_train,x_test,y_train,y_test =train_test_split(x,y,random_state=0,test_size=0.3)

rfc.fit(x_train,y_train)
y_rfc=rfc.predict(x_test)

logr.fit(x_train,y_train)
y_logr=logr.predict(x_test)

gbc.fit(x_train,y_train)
y_gbc=gbc.predict(x_test)

dtc.fit(x_train,y_train)
y_dtc=dtc.predict(x_test)

svm.fit(x_train,y_train)
y_svm=svm.predict(x_test)

nn.fit(x_train,y_train)
y_nn=nn.predict(x_test)

nb.fit(x_train,y_train)
y_nb=nb.predict(x_test)

print("Random Forest:", accuracy_score(y_test,y_rfc))
print("Logistic Regression:", accuracy_score(y_test,y_logr))
print("Gradient Boosting:", accuracy_score(y_test,y_gbc))
print("Decision Tree:", accuracy_score(y_test,y_dtc))
print("Support Vector Machine:", accuracy_score(y_test,y_svm))
print("Artificial Neural Network:", accuracy_score(y_test,y_nn))
print("Naive Bayes:", accuracy_score(y_test,y_nb))

'''
OUTPUT:
Random Forest: 0.9777777777777777
Logistic Regression: 0.9777777777777777
Gradient Boosting: 0.9777777777777777
Decision Tree: 0.9777777777777777
Support Vector Machine: 0.9777777777777777
Artificial Neural Network: 0.24444444444444444
Naive Bayes: 0.6
'''
