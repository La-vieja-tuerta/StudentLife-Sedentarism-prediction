from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import SGD

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

X = pd.read_pickle('Xsamples.pkl')
y = pd.read_pickle('ysamples.pkl')
#X.drop(columns='wifiChanges', inplace=True)

label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)
y = to_categorical(y_int)

X.loc[X['beforeNextDeadline'] > 0, 'beforeNextDeadline'] = np.log(X.loc[X['beforeNextDeadline'] > 0, 'beforeNextDeadline'])
X.loc[X['afterLastDeadline'] > 0, 'afterLastDeadline'] = np.log(X.loc[X['afterLastDeadline'] > 0, 'afterLastDeadline'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
#print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
numeric_cols = ['cantConversation', 'beforeNextDeadline', 'afterLastDeadline']

ss = StandardScaler()
X_train[numeric_cols] = ss.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = ss.transform(X_test[numeric_cols])

size = X.shape[1]
# Initialize the constructor
model = Sequential([
    Dense(300, activation='relu', input_shape=(size,)),
    Dense(300, activation='relu'),
    Dense(300, activation='relu'),
    Dense(300, activation='relu'),
    Dense(2, activation='softmax')
])
sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])


h = model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=2,
          validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

model.summary()
