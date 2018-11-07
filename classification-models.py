from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import SGD
import pandas as pd
import numpy as np
from utilfunction import *
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt

df = pd.read_pickle('sedentarismdata.pkl')
df = delete_user(df,52)
df = METcalculation(df)
df = makeSedentaryClasses(df)
df = makeDummies(df)
df = shift_hours(df,1, 'classification')
df.drop(['audiomajor', 'hourofday', 'stationaryCount', 'walkingCount', 'runningCount',
         'latitudeMean', 'longitudeMean',
         'latitudeMedian', 'longitudeMedian',
         'latitudeStd', 'longitudeStd'
         ], axis=1, inplace=True)

X, y = get_X_y_classification(df, True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

numeric_cols = ['cantConversation', 'wifiChanges',
                'silenceCount', 'voiceCount', 'noiseCount', 'unknownAudioCount',
                'remainingminutes', 'pastminutes']
#                'latitudeMean', 'longitudeMean',
#                'latitudeMedian', 'longitudeMedian',
#                'latitudeStd', 'longitudeStd']

ss = StandardScaler()
X_train.loc[:, numeric_cols] = ss.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = ss.transform(X_test[numeric_cols])


'''
#codigo para usar oversampling
columns = X.columns
sm = SMOTE(random_state=12, ratio='all')
X_train, y_train = sm.fit_sample(X_train, y_train)
X_train = pd.DataFrame(X_train, columns=columns)
y_train = pd.Series(y_train)

clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=350)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
print(classification_report(y_test, clf.predict(X_test)))
'''
size = X.shape[1]


model = Sequential([
Dense(1024, activation='relu', input_dim=size),
Dense(1024, activation='relu', input_dim=size),
Dense(1024, activation='relu', input_dim=size),
Dense(1024, activation='relu', input_dim=size),
Dense(1024, activation='relu', input_dim=size),
Dense(1024, activation='relu', input_dim=size),
Dense(512, activation='relu', input_dim=size),
Dropout(.4),
Dense(256, activation='relu'),
Dropout(.2),
Dense(128, activation='relu'),
Dropout(.1),
Dense(64, activation='relu'),
Dropout(.1),
Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])

#clf = LogisticRegression(solver='liblinear', max_iter=400)

h = model.fit(X_train, y_train, epochs=10, batch_size=512, verbose=2,
          validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

y_pred =  y_pred > 0.5
y_pred = y_pred.astype('int')


print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(classification_report(y_test, DummyClassifier(strategy='most_frequent', random_state=7)
                           .fit(X_train,y_train).predict(X_test)))
#model.summary()
plt.close()
cols = X.columns
nums = np.arange(1,40)
a = clf.coef_
plt.plot(nums,a.reshape(-1,1))
plt.xticks(nums, cols, rotation='vertical')
plt.grid(True)
plt.show()