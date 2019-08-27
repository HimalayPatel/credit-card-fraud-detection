import pandas as pd
import numpy as np
import keras

np.random.seed(2)
data = pd.read_csv("C:\\Users\\himal\\Desktop\\Machine Learning Practicals\\P6- Credit Card fraud detection\\P39-Credit-Card-Fraud\Dataset\\creditcard.csv")
# print(data.head())
from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'],axis=1)
# print(data.head())
data = data.drop(['Time'],axis=1)
# print(data.head())
X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']
# print(X.head())
# print(y.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
# print(X_train.shape)
# print(X_test.shape)

from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier()
dt_model.fit(X_train,y_train.values.ravel())
y_pred=dt_model.predict(X_test)
score=dt_model.score(X_test,y_test)
print(score)

# Confusion Matrix code from sklearn
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Confusion Matrix of Test Dataset
y_pred = dt_model.predict(X_test)
y_test = pd.DataFrame(y_test)
cnf_matrix = confusion_matrix(y_test, y_pred.round())
print(cnf_matrix)
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()
# Confusion Matrix of whole Dataset
y_pred = dt_model.predict(X)
y_expected=pd.DataFrame(y)
cnf_matrix=confusion_matrix(y_expected,y_pred.round())
print(cnf_matrix)
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
