import pandas as pd
import numpy as np

#prep
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler, QuantileTransformer
from sklearn.svm import SVC, LinearSVC
#models
from sklearn.linear_model import LogisticRegression, SGDClassifier

#validation libraries
from sklearn.metrics import classification_report
np.random.seed(7)

s = pd.read_pickle('sedentarismdata.pkl')
s = s.head(41872)
s['slevel'] = pd.qcut(s['isSedentary'], 3, duplicates='drop', labels=['less sedentary','sedentary'])

s = s.sort_values('isSedentary')

s['slevel'] = ''
s.loc[s['isSedentary'] > 0.9999, 'slevel'] = 'very sedentary'
s.loc[s['isSedentary'].between(0.9052, 0.9999), 'slevel'] = 'sedentary'
s.loc[s['isSedentary'] < 0.9052, 'slevel'] = 'less sedentary'

s = s.drop(columns=['isSedentary', 'hourofday',
                    'audiomajor'])



#set type of numeric and categorical columns
numeric_cols = ['cantConversation', 'beforeNextDeadline', 'afterLastDeadline', 'wifiChanges']
for col in numeric_cols:
    s[col] = s[col].astype('float')

categorical_cols = [x for x in s.columns if x not in numeric_cols]
for col in categorical_cols:
    s[col] = s[col].astype('category')

swithdummies = pd.get_dummies(s, columns=['dayofweek', 'activitymajor', 'partofday'])

#se hace el shift para que el y de cada x corresponda al nivel de sedentarismo de una hora posterior
swithdummies = swithdummies.sort_index()

swithdummies['slevel'] = swithdummies['slevel'].shift(-1)

#se descartan las filas de x que correspondan a una hora sobre la que no haya informacion
# sobre el nivel de sedentarismo de la hora siguiente

for ind, row in swithdummies.iterrows():
    if not (ind[0], ind[1] + pd.DateOffset(hours=1)) in swithdummies.index:
        swithdummies.loc[(ind[0], ind[1])] = np.nan
swithdummies = swithdummies.dropna()

features = feature_cols = [col for col in swithdummies.columns if 'slevel' != col]
X = swithdummies[features]
y = swithdummies['slevel']

X.to_pickle('Xsamples.pkl')
y.to_pickle('ysamples.pkl')




X = pd.read_pickle('Xsamples.pkl')
y = pd.read_pickle('ysamples.pkl')
#X.drop(columns='wifiChanges', inplace=True)

X.loc[X['beforeNextDeadline'] > 0, 'beforeNextDeadline'] = np.log(X.loc[X['beforeNextDeadline'] > 0, 'beforeNextDeadline'])
X.loc[X['afterLastDeadline'] > 0, 'afterLastDeadline'] = np.log(X.loc[X['afterLastDeadline'] > 0, 'afterLastDeadline'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

ss = StandardScaler()
X_train[numeric_cols] = ss.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = ss.transform(X_test[numeric_cols])


cls = LinearSVC(C=1000, verbose=True)
cls.fit(X_train, y_train)

y_true, y_pred = y_test, cls.predict(X_test)
print(classification_report(y_true, y_pred))

clf = SGDClassifier(max_iter=10000, verbose=True)

clf.fit(X_train, y_train)

y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))

logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', max_iter=1000)
logreg.fit(X_train, y_train)
print(classification_report(logreg.predict(X_test.values), y_test.values))

