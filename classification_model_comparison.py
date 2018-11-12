from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from utilfunction import *
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.wrappers.scikit_learn import KerasClassifier
import numpy
numpy.random.seed(7)

def show_metric(title, ylabel, labels, data):
    users = np.arange(1, 49)
    userslabel = df.index.get_level_values(0).drop_duplicates()
    plt.close()
    for d in data:
        plt.scatter(users, d)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('user')
    plt.legend(labels,
               loc='lower right')
    plt.xticks(users, userslabel, rotation='vertical')
    plt.grid(True)
    plt.show()


estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=256, verbose=2)

clf = LogisticRegression(solver='liblinear', max_iter=400)
model = create_model(estimator)
df = pd.read_pickle('sedentarismdata.pkl')
df = delete_user(df,52)
df = METcalculation(df)
df = makeSedentaryClasses(df)
df = makeDummies(df)
df = shift_hours(df,1, 'classification')
df.drop(['hourofday'],
        axis=1, inplace=True)
#stationaryLevel, walkingLevel, runningLevel
precisionLOGOwithACNN, recallLOGOwithACNN = live_one_out_classificationNN(df, True)
precisionPUwithAC, recallPUwithAC = per_user_classification(df, model, True)
precisionPUwithoutAC, recallPUwithoutAC = per_user_classification(df, model, False)

precisionLOGOwithoutACNN, recallLOGOwithoutACNN = live_one_out_classificationNN(df, False)
precisionLOGOwithAC, recallLOGOwithAC = live_one_out_classification(df, model, True)
precisionLOGOwithoutAC, recallLOGOwithoutAC = live_one_out_classification(df, model, False)

show_metric('Model precision',
            'Precision',
            ['precisionLOGOwithAC', 'precisionLOGOwithoutAC'],
            [precisionLOGOwithAC, precisionPUwithAC])

show_metric('Model recall',
            'Recall',
            ['recallLOGOwithAC', 'recallLOGOwithoutAC'],
            [recallLOGOwithAC, recallPUwithAC])

show_metric('Model precision',
            'Precision',
            ['precisionPUwithAC', 'precisionPUwithoutAC'],
            [precisionPUwithAC, precisionPUwithoutAC])

show_metric('Model recall',
            'Recall',
            ['recallPUwithAC', 'recallPUwithoutAC'],
            [recallPUwithAC, recallPUwithoutAC])



show_metric('Model precision',
            'Precision',
            ['precisionLOGOwithAC', 'recallLOGOwithACNN'],
            [precisionLOGOwithACNN, recallLOGOwithACNN])

show_metric('Model recall',
            'Recall',
            ['recallLOGOwithAC', 'recallLOGOwithoutAC'],
            [recallLOGOwithACNN, recallLOGOwithoutAC])

print(np.mean(precisionLOGOwithAC))
print(np.mean(precisionLOGOwithoutAC))
print(np.mean(precisionPUwithAC))
print(np.mean(precisionPUwithoutAC))


