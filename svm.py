import pandas as pd
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import *
from sklearn import svm
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.svm import SVC
import itertools
def plot_confusion_matrix(cm,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, rotation=45)
    # plt.yticks(tick_marks)
    normalize = True
    if normalize:
        cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(round(cm[i, j], 2)) + '%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

training_data_df = pd.read_csv("features500.csv")
X = training_data_df.iloc[:, 1:].values
y = training_data_df.iloc[:, 0].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# kfold = KFold(3, True, 1)
# for train, test in kfold.split():
# 	print('train: %s, test: %s' % (data[train], data[test]))
_svm = svm.SVC(gamma=2, C=1)
_svm.fit(X_train,np.ravel(y_train))
#plt.scatter(y_test,_svm.predict(X_test))
#plt.show()
y_pred=_svm.predict(X_test)
print(_svm.score(X_test, np.ravel(y_test)))


# scores = cross_val_score(SVC(kernel="linear", C=0.025), X, y, cv=10)
cnf_matrix = confusion_matrix(y_test, y_pred)
a,b = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
tp=a[0]
fn=a[1]
fp=b[0]
tn=b[1]
print("true positive :"+str(tp))
print("False positive :"+str(fp))
print("False negative :"+str(fn))
print("True negative :"+str(tn))

sensitivity=tp/(tp+fn)
specificity=tn/(tn+fp)
p=tp/(tp+fp)
r=tp/tp+fn
a=(tp+tn)/(tp+tn+fp+fn)
f1=2*(p*r)/(p+r)
print("sensitivity :"+str(sensitivity))
print("specificity :"+str(specificity))
print("precision :"+str(p))
print("recall :"+str(r))
print("accuracy :"+str(a))
print("F1 :"+str(f1))
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, title=('Confusion matrix of knn'))
# mean = scores.mean()
# stdev = scores.std()
plt.show()

print (classification_report(y_test, y_pred) )
# print("[Results For ] Mean: ",mean," Std Dev: ",stdev)