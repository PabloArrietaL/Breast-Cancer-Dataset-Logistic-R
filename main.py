from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from random import randrange
from time import time
import sklearn.metrics as m
import pandas as pd
import numpy as np

initial = time()
dataSet = pd.read_csv("wdbc.txt",delimiter=',')

X, y = dataSet.values[:,2:], dataSet.values[:,1]
np.place(y, y=='M',1)
np.place(y, y=='B',0)
y=y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0, stratify=y)
c = [randrange(1, 10000, 100)/float(1000) for i in range(100) ]
c = list(set(c))

Logis1 = [LogisticRegression(C=i, solver="lbfgs", max_iter=1000).fit(X_train,y_train) for i in c]
cros = [cross_val_score(i,X_train,y_train,cv=10).mean() for i in Logis1]
LR = LogisticRegression(C=c[cros.index(max(cros))], solver="lbfgs", max_iter=1000).fit(X_train,y_train)
Model = LR.predict(X_test)

f1 = m.f1_score(y_test,Model)
recall = m.recall_score(y_test,Model)
accuracy = m.accuracy_score(y_test,Model)
precision = m.precision_score(y_test,Model)
tn, fp, fn, tp = m.confusion_matrix(y_test,Model).ravel()
specificity = tn/float(tn+fp)
final = time()
print("\n\t   CONFUSION MATRIX")
print("         Negative     Positive")
print("Negative   {0}           {1}".format(tn,fp))
print("Positive   {0}            {1}".format(fn,tp))
print("\nF1-score: {0}\nRecall: {1}\nAccuracy: {2}\nPrecision: {3}\nSpecificity: {4}".format(f1,recall,accuracy,precision,specificity))
print("\nExecution time: "+str(round(final-initial,2))+" seconds")
