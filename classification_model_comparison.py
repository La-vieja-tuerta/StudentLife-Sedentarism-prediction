from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from utilfunction import *
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

from numpy.random import seed
seed(7)

numeric_cols = ['cantConversation', 'wifiChanges', 'stationaryCount', 'walkingCount',
                'runningCount', 'silenceCount', 'voiceCount', 'noiseCount',
                'unknownAudioCount', 'pastminutes','remainingminutes']

transformer = ColumnTransformer([('scale', StandardScaler(), numeric_cols)],
                                remainder='passthrough')

clf = LogisticRegression(solver='liblinear', max_iter=400)

model = make_pipeline(transformer, clf)

df = pd.read_pickle('sedentarismwithoutdummies')
df.drop(['audiomajor', 'hourofday'], axis=1, inplace=True)
df = makeDummies(df)
df = METcalculation(df)
df = makeSedentaryClasses(df)
df = shift_hours(df,1, 'classification')


precisionPUwithAC, recallPUwithAC = per_user_classification(df, model, True)
precisionLOGOwithAC, recallLOGOwithAC = live_one_out_classification(df, model, True)

precisionPUwithoutAC, recallPUwithoutAC = per_user_classification(df, model, False)
precisionLOGOwithoutAC, recallLOGOwithoutAC = live_one_out_classification(df, model, False)

users = np.arange(1,50)
plt.close()
plt.scatter(users, precisionPUwithAC, label='precisionPUwithAC')
plt.scatter(users, precisionLOGOwithAC, label='precisionLOGOwithAC')
plt.scatter(users, precisionPUwithoutAC, label='precisionPUwithoutAC')
plt.scatter(users, precisionLOGOwithoutAC, label='precisionLOGOwithoutAC')

plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('user')
plt.legend(['precisionPUwithAC', 'precisionLOGOwithAC', 'precisionPUwithoutAC', 'precisionLOGOwithoutAC'],
           loc='lower right')
plt.show()

plt.scatter(users, recallPUwithAC, label='recallPUwithAC')
plt.scatter(users, recallLOGOwithAC, label='recallLOGOwithAC')
plt.scatter(users, recallPUwithoutAC, label='recallPUwithoutAC')
plt.scatter(users, recallLOGOwithoutAC, label='recallLOGOwithoutAC')

plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('user')
plt.legend(['recallPUwithAC', 'recallLOGOwithAC', 'recallPUwithoutAC', 'recallLOGOwithoutAC'],
           loc='lower right')
plt.xticks(users, users, rotation='vertical')
plt.grid(True)
plt.show()


