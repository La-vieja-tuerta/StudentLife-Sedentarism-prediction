from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from utilfunction import *
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.wrappers.scikit_learn import KerasClassifier

from numpy.random import seed
seed(7)

def show_metric(title, ylabel, labels, data):
    users = np.arange(1, 50)
    plt.close()
    for d in data:
        plt.scatter(users, d)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('user')
    plt.legend(labels,
               loc='lower right')
    plt.xticks(users, users, rotation='vertical')
    plt.grid(True)
    plt.show()


numeric_cols = ['cantConversation', 'wifiChanges', 'stationaryCount', 'walkingCount',
                'runningCount', 'silenceCount', 'voiceCount', 'noiseCount',
                'unknownAudioCount', 'pastminutes','remainingminutes']

transformer = ColumnTransformer([('scale', StandardScaler(), numeric_cols)],
                                remainder='passthrough')

estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=256, verbose=2)

clf = LogisticRegression(solver='liblinear', max_iter=400)
clf2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),
                         algorithm="SAMME",
                         n_estimators=200)
model = make_pipeline(transformer, clf)
df = pd.read_pickle('sedentarisdata.pkl')
df.drop(['audiomajor', 'hourofday'], axis=1, inplace=True)
df = makeDummies(df)
df = METcalculation(df)
df = makeSedentaryClasses(df)
df = shift_hours(df,1, 'classification')

precisionLOGOwithAC, recallLOGOwithAC = live_one_out_classificationNN(df, True)
precisionPUwithAC, recallPUwithAC = per_user_classification(df, model, True)

precisionPUwithoutAC, recallPUwithoutAC = per_user_classification(df, model, False)
precisionLOGOwithoutAC, recallLOGOwithoutAC = live_one_out_classificationNN(df, False)

show_metric('Model precision with NN',
            'Precision',
            ['precisionLOGOwithAC', 'precisionLOGOwithoutAC'],
            [precisionLOGOwithAC, precisionLOGOwithoutAC])

show_metric('Model recall with NN',
            'Recall',
            ['recallLOGOwithAC', 'recallLOGOwithoutAC'],
            [recallLOGOwithAC, recallLOGOwithoutAC])
