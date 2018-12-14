import pandas as pd
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
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

#
# def plot_axis(axis, x, y, title):
#     axis.plot(x, y)
#     axis.set_title(title)
#     axis.xaxis.set_visible(False)
#     axis.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
#     axis.set_xlim([min(x), max(x)])
#     axis.grid(True)
#
# def plotactivity(act, data):
#     fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
#     plot_axis(ax0, data['timespace'], data['gccx'], 'x-axis')
#     plot_axis(ax1, data['timespace'], data['gccy'], 'y-axis')
#     plot_axis(ax2, data['timespace'], data['gccz'], 'z-axis')
#     plt.subplots_adjust(hspace=0.2)
#     fig.suptitle(act)
#     plt.subplots_adjust(top=0.90)
#     plt.show()


column_names = ['accx', 'accy', 'accz', 'gccx', 'gccy', 'gccz', 'activity','timspace']

training_data_df = pd.read_csv("sales_data_training_scaledwith time.csv")

# for activity in np.unique(training_data_df["activity"]):
#     subset = training_data_df[training_data_df["activity"] == 0.0][:1080]
#     plotactivity(activity,subset)
training_data_df=training_data_df.drop('timespace',axis=1)
x=training_data_df.drop('activity',axis=1).values

y=training_data_df[['activity']].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# Define the model
model = Sequential()
model.add(Dense(50, input_dim=6, activation='relu'))
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
    epochs=1,
    shuffle=True,
    validation_data=(X_test,y_test),
    verbose=2,
    callbacks = [logger]
 )

test_error_rate = model.evaluate(X_test, y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
# Save the model to disk
model.save("trained_model.h5")
y_pred=model.predict(X_test)
print("Model saved to disk.")
# cnf_matrix = confusion_matrix(y_test, y_pred.round())
# print(cnf_matrix)
# if(((y_pred.round()!=0)|(y_pred.round()!=1)).any()):
#     y_pred=1
# elif((y_pred.round()==0).any()):
#     y_pred=0
# else:
#     y_pred=1
#
# cnf_matrix = confusion_matrix(y_test,y_pred )
# a,b = confusion_matrix(y_test, y_pred)
# print(cnf_matrix)
# tp=a[0]
# fn=a[1]
# fp=b[0]
# tn=b[1]
# print("true positive :"+str(tp))
# print("False positive :"+str(fp))
# print("False negative :"+str(fn))
# print("True negative :"+str(tn))
#
# sensitivity=tp/(tp+fn)
# specificity=tn/(tn+fp)
# p=tp/(tp+fp)
# r=(tp+tn)/(tp+tn+fp+fn)
# f1=2*(p*r)/(p+r)
# print("sensitivity :"+str(sensitivity))
# print("specificity :"+str(specificity))
# print("precision :"+str(p))
# print("accuracy :"+str(r))
# print("F1 :"+str(f1))
# np.set_printoptions(precision=2)
# plt.figure()
# plot_confusion_matrix(cnf_matrix, title=('Confusion matrix of knn'))
#
# # mean = scores.mean()
# # stdev = scores.std()
# print (classification_report(y_test, y_pred.round()))
# plt.show()
# # print("[Results For ] Mean: ",mean," Std Dev: ",stdev)
# print (classification_report(y_test, (y_pred>1).astype(int) ))