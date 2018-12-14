import pandas as pd
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.svm import SVC
import itertools


def feature_normalization(dataset):
    # network to work well.
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Scale both the training inputs and outputs
    scaled_training = scaler.fit_transform(dataset)
    return scaled_training

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
        np.seterr(divide='ignore', invalid='ignore')
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

model = Sequential()
model.add(Dense(50, input_dim=64, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile( loss='mean_squared_error', optimizer="adam", metrics=['accuracy'] )

# Create a TensorBoard logger
logger = TensorBoard(
    log_dir='logs',
    write_graph=True,
    histogram_freq=5
)
model.fit(
    X_train,
    y_train,
    epochs=50,
    shuffle=True,
    validation_data=(X_test,y_test),
    verbose=2,
    callbacks = [logger]
 )

test_error_rate = model.evaluate(X_test, y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
# Save the model to disk
y_pred=model.predict(X_test)
model.save("trained_model.h5")
print("Model saved to disk.")
print((y_pred>1).astype(int))
# scores = cross_val_score(model, X, y, cv=10)

cnf_matrix = confusion_matrix(y_test,(y_pred>1).astype(int) )
a,b = confusion_matrix(y_test, (y_pred>1).astype(int))
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
r=tp/(tp+fn)
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
plot_confusion_matrix(cnf_matrix, title=('Confusion matrix of CNN'))

# mean = scores.mean()
# stdev = scores.std()
print (classification_report(y_test, y_pred.round()))
plt.show()
# print("[Results For ] Mean: ",mean," Std Dev: ",stdev)
print (classification_report(y_test, (y_pred>1).astype(int) ))