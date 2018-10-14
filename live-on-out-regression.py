from utilfunction import gen_live_one_out, get_X_y_regression
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import KFold
from sklearn import metrics

seed = 7
np.random.seed(seed)
numeric_cols = ['cantConversation', 'beforeNextDeadline', 'afterLastDeadline', 'wifiChanges',
                'stationaryCount', 'walkingCount', 'runningCount', 'silenceCount', 'voiceCount', 'noiseCount',
                'unknownAudioCount']

s = pd.read_pickle('sedentarism.pkl')

transformer = ColumnTransformer([('transformer', StandardScaler(),
                                  numeric_cols)],
                                remainder='passthrough')

model = make_pipeline(transformer, linear_model.LinearRegression())

mse = []
variance = []
i = 0
for train,test in gen_live_one_out(s):
    X_train, y_train = get_X_y_regression(train)
    X_test, y_test = get_X_y_regression(test)
    kfold = KFold(n_splits=10, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse.append(mean_squared_error(y_test, y_pred))
    variance.append(r2_score(y_test, y_pred))
    if i%10 == 0:
        print('modelos sobre usuario ', i, ' finalizado.')
    i += 1

fig, (ax1,ax2) = plt.subplots(2, 1)
ax1.plot(mse)
ax1.set_title('mse')
ax2.plot(variance)
ax2.set_title('variance')
fig.show()
