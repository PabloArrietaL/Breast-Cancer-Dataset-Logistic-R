from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from random import randrange
import sklearn.metrics as m

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
c = [randrange(1, 1000) for i in range(20) ]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0, stratify=y)
X_train_Nor = StandardScaler().fit_transform(X_train[:])
X_test_Nor = StandardScaler().fit_transform(X_test[:])

svm = [SVC(C=i,gamma='scale').fit(X_train_Nor,y_train) for i in c]
cros = [cross_val_score(i,X_train_Nor,y_train,cv=10,n_jobs=-1).mean() for i in svm]

svm_f = SVC(C=cros[cros.index(max(cros))],gamma='scale').fit(X_train_Nor,y_train)
model = svm_f.predict(X_test_Nor) 

f1 = m.f1_score(y_test,model)
recall = m.recall_score(y_test,model)
accuracy = m.accuracy_score(y_test,model)
precision = m.precision_score(y_test,model)
tn, fp, fn, tp = m.confusion_matrix(y_test,model).ravel()
specificity = tn/float(tn+fp)

print("\n\t   CONFUSION MATRIX")
print("         Negative     Positive")
print("Negative   {0}           {1}".format(tn,fp))
print("Positive   {0}           {1}".format(fn,tp))
print("\nF1-score: {0}\nRecall: {1}\nAccuracy: {2}\nPrecision: {3}\nSpecificity: {4}".format(f1,recall,accuracy,precision,specificity))
