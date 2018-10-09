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

from imblearn.over_sampling import SMOTE


rng = np.random

s = pd.read_pickle('sedentarismdata.pkl')
s = s.sort_values('isSedentary')
'''
s = s.head(41397) #numero magico
s['slevel'] = pd.qcut(s['isSedentary'], 3, labels=['less sedentary', 'sedentary', 'very sedentary'])
'''

s['slevel'] = ''
s.loc[s['isSedentary'] > 0.9999, 'slevel'] = 'very sedentary'
s.loc[s['isSedentary'].between(0.9052, 0.9999), 'slevel'] = 'sedentary'
s.loc[s['isSedentary'] < 0.9052, 'slevel'] = 'less sedentary'

s = s.drop(columns=['isSedentary', 'audiomajor'])
#set type of numeric and categorical columns
numeric_cols = ['cantConversation', 'beforeNextDeadline', 'afterLastDeadline', 'hourofday', 'wifiChanges']
for col in numeric_cols:
    s[col] = s[col].astype('float')

categorical_cols = [x for x in s.columns if x not in numeric_cols]
for col in categorical_cols:
    s[col] = s[col].astype('category')

swithdummies = pd.get_dummies(s.copy(), columns=['dayofweek', 'activitymajor', 'partofday'])
#se hace el shift para que el y de cada x corresponda al nivel de sedentarismo de una hora posterior
swithdummies = swithdummies.sort_index()
swithdummies['slevel'] = swithdummies['slevel'].shift(-1)

#se descartan las filas de x que correspondan a una hora sobre la que no haya informacion
# sobre el nivel de sedentarismo de la hora siguiente

for ind, row in swithdummies.iterrows():
    if not (ind[0], ind[1] + pd.DateOffset(hours=1)) in swithdummies.index:
        swithdummies.loc[(ind[0], ind[1])] = np.nan
swithdummies = swithdummies.dropna()

features = [col for col in swithdummies.columns if 'slevel' != col]
X = swithdummies[features]
y = swithdummies['slevel']

# a continuacion codigo para usar SMOTE
'''
columns = X.columns
sm = SMOTE(random_state=12, ratio='all')
X, y = sm.fit_sample(X, y)
X = pd.DataFrame(X, columns=columns)
y = pd.Series(y)
'''

X.to_pickle('Xsamples.pkl')
y.to_pickle('ysamples.pkl')