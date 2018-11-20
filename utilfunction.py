from collections import Counter
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization,Activation
from keras.wrappers.scikit_learn import KerasClassifier
from haversine import haversine

from sklearn.metrics import confusion_matrix
from numpy.random import seed
seed(7)


def createSensingTable(path):
    df = pd.read_csv(path + '00' + '.csv', index_col=False)
    df['userId'] = '00'
    for a in range(1, 60):
        userId = '0' + str(a) if a < 10 else str(a)
        try:
            aux = pd.read_csv(path + userId + '.csv', index_col=False)
            aux['userId'] = a
            df = df.append(aux)
        except:
            pass
    return df


def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def get_user_data(data, userId):
    try:
        return data.loc[data.index.get_level_values(0) == userId].copy()
    except KeyError:
        print('El usuario ', userId, ' no existe.')

def get_X_y_regression(df):
    dfcopy = df.copy()
    features = [col for col in dfcopy.columns if 'slevel' != col]
    return dfcopy[features].reset_index(drop=True), dfcopy['slevel'].reset_index(drop=True)

def makeSedentaryClasses(df):
    dfcopy = df.copy()
    dfcopy['sclass'] = ''
    dfcopy.loc[df['slevel'] >= 1.5, 'sclass'] = 0.0  # 'sedentary'
    dfcopy.loc[df['slevel'] < 1.5, 'sclass'] = 1.0  # 'not sedentary'
    dfcopy['actualClass'] = dfcopy['sclass']
    dfcopy.drop(['slevel'], inplace=True, axis=1)
    return dfcopy

def get_X_y_classification(df, withActualClass):
    dfcopy = df.copy()
    if not withActualClass:
        dfcopy.drop(['actualClass'], inplace=True, axis=1)
    features = [col for col in dfcopy.columns if 'sclass' != col]
    return dfcopy[features].reset_index(drop=True), dfcopy['sclass'].reset_index(drop=True)


def per_user_regression(df, model):
    print('per_user_regression')
    dfcopy = df.copy()
    mse = []
    for userid in df.index.get_level_values(0).drop_duplicates():
        X, y = get_X_y_regression(get_user_data(dfcopy, userid))
        kfold = StratifiedKFold(n_splits=10, random_state=seed)
        results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
        mse.append(-results.mean())
        if userid % 10 == 0:
            print('modelos sobre usuario ', userid, ' finalizado.')
    return mse

def live_one_out_regression(df, model):
    print('live_one_out_regression')
    dfcopy = df.copy()
    mse = []
    i = 0
    logo = LeaveOneGroupOut()
    groups = df.index.get_level_values(0)
    X, y = get_X_y_regression(dfcopy)
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse.append(mean_squared_error(y_test, y_pred))
        if i % 10 == 0:
            print('modelos sobre usuario ', i, ' finalizado.')
        i += 1
    return mse


def per_user_classification(df, model, withActualClass):
    print('per_user_classification')
    dfcopy = df.copy()
    scoring = ['f1_weighted']
    f1 = []
    kfold = StratifiedKFold(n_splits=10, random_state=seed)
    for userid in df.index.get_level_values(0).drop_duplicates():
        X, y = get_X_y_classification(get_user_data(dfcopy, userid), withActualClass)
        results = cross_validate(model, X, y, cv=kfold, scoring=scoring)
        f1.append(results['test_f1_weighted'].mean())
        if userid % 10 == 0:
            print('modelos sobre usuario ', userid, ' finalizado.')
    return f1

def live_one_out_classification(df, model, withActualClass):
    dfcopy = df.copy()
    print('live_one_out_classification')
    i = 0
    f1 = []
    logo = LeaveOneGroupOut()
    groups = df.index.get_level_values(0)
    X, y = get_X_y_classification(dfcopy, withActualClass)
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1.append(f1_score(y_test, y_pred, average='weighted'))
        if i % 10 == 0:
            print('modelos sobre usuario ', i, ' finalizado.')
        i += 1
    return f1

def live_one_out_classificationNN(df, withActualClass):
    dfcopy = df.copy()
    print('live_one_out_classification')
    i = 0
    precision = []
    recall = []
    logo = LeaveOneGroupOut()
    groups = df.index.get_level_values(0)
    X, y = get_X_y_classification(dfcopy, withActualClass)
    y = to_categorical(y)
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = create_model(KerasClassifier(build_fn=baseline_model, input_dim=X.shape[1],
                                             epochs=10, batch_size=256, verbose=0,
                                             validation_data=(X_test, y_test)))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_test = np.argmax(y_test, axis=1)
        precision.append(precision_score(y_test, y_pred, average='weighted'))
        recall.append(recall_score(y_test, y_pred, average='weighted'))
        if i % 10 == 0:
            print('modelos sobre usuario ', i, ' finalizado.')
        i += 1
    return precision, recall


def shift_hours(df, n, modelType):
    print('Shifting ', n, 'hours.')
    dfcopy = df.copy().sort_index()
    for ind, row in dfcopy.iterrows():
        try:
            if modelType == 'regression':
                dfcopy.at[(ind[0], ind[1]),'slevel'] = dfcopy.at[(ind[0], ind[1] + pd.DateOffset(hours=n)), 'slevel']
            elif modelType == 'classification':
                dfcopy.at[(ind[0], ind[1]),'sclass'] = dfcopy.at[(ind[0], ind[1] + pd.DateOffset(hours=n)), 'sclass']
        except KeyError:
            if modelType == 'regression':
                dfcopy.at[(ind[0], ind[1]), 'slevel'] = np.nan
            elif modelType == 'classification':
                dfcopy.at[(ind[0], ind[1]), 'sclass'] = np.nan
    print(dfcopy.isna().sum())
    dfcopy.dropna(inplace=True)
    return dfcopy

def create_model(clf):
    numeric_cols = ['cantConversation', 'wifiChanges',
                    'silenceLevel', 'voiceLevel', 'noiseLevel',
                    'remainingminutes', 'pastminutes',
                    'distanceTraveled', 'locationVariance']
    transformer = ColumnTransformer([('scale', MinMaxScaler(), numeric_cols)],
                                    remainder='passthrough')
    return make_pipeline(transformer, clf)


def METcalculation(df, metValues=(1.3,5,8.3)):
    dfcopy = df.copy()
    metLevel = (dfcopy['stationaryLevel'] * metValues[0] +
                dfcopy['walkingLevel'] * metValues[1] +
                dfcopy['runningLevel'] * metValues[2])
    dfcopy['slevel'] = metLevel
    return dfcopy


def makeDummies(df):
    dfcopy = df.copy()
    categorical_cols = ['partofday', 'dayofweek', 'activitymajor']
    for col in categorical_cols:
        dfcopy[col] = dfcopy[col].astype('category')
    for col in set(df.columns) - set(categorical_cols):
        dfcopy[col] = dfcopy[col].astype('float')
    dummies = pd.get_dummies(dfcopy.select_dtypes(include='category'))
    dfcopy.drop(categorical_cols, inplace=True, axis=1)
    return pd.concat([dfcopy, dummies], axis=1, sort=False)

def baseline_model():
    estimator = Sequential([
    Dense(256,input_dim=31,kernel_initializer='uniform', kernel_regularizer='l2',use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(128, kernel_initializer='uniform', kernel_regularizer='l2',use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(64, kernel_initializer='uniform',kernel_regularizer='l2',use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(32, kernel_initializer='uniform', kernel_regularizer='l2',use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(1, activation='sigmoid')
    ])
    estimator.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])
    return estimator

def delete_user(df,user):
    return df.copy().loc[df.index.get_level_values(0)!=user]

def get_total_harversine_distance_traveled(x):
    d = 0.0
    samples = x.shape[0]
    for i in np.arange(0,samples):
        try:
            d += haversine(x.iloc[i,:].values, x.iloc[i+1,:].values)
        except IndexError:
            pass
    return d

def delete_sleep_hours(df):
    dfcopy = df.copy()
    return dfcopy.loc[(dfcopy['slevel'] >= 1.5) | (dfcopy['partofday'] != 'night')]