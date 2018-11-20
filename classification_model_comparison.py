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
               loc='upper right')
    plt.xticks(users, userslabel, rotation='vertical')
    plt.ylim(0.5,1)
    plt.grid(True)
    plt.show()

df = pd.read_pickle('sedentarismdata.pkl')
df = delete_user(df,52)
df = METcalculation(df)
#df = delete_sleep_hours(df)
df = makeSedentaryClasses(df)
df = makeDummies(df)
df = shift_hours(df,1, 'classification')
df.drop(['hourofday'],
        axis=1, inplace=True)

estimator = KerasClassifier(build_fn=baseline_model, epochs=30, batch_size=512, verbose=0)
modelnn = create_model(estimator)
clf = LogisticRegression(solver='liblinear', max_iter=400, class_weight='balanced')
model = create_model(clf)

#f1_p_nn = per_user_classification(df, modelnn, True)
f1_p_logreg = per_user_classification(df, model, True)

f1_imp_nn = live_one_out_classification(df, modelnn, True)
f1_imp_logreg = live_one_out_classification(df, model, True)



b = df.groupby(df.index.get_level_values(0))['sclass'].apply(lambda x : np.sum(x==1)).values
c = df.groupby(df.index.get_level_values(0))['sclass'].count().values
b = b/c

"""
show_metric('Model F1-score for imersonal models ',
            'F1-score',
            ['nn', 'logreg'],
            [f1_imp_nn, f1_imp_logreg])
"""

show_metric('F1-score comparison between impersonal and personal models ',
            'F1-score',
            ['f1_imp', 'f1_p'],
            [f1_imp_logreg, f1_p_logreg])

print(np.mean(f1_imp_logreg))
print(np.std(f1_imp_logreg))

np.mean(np.abs(np.subtract(f1_imp_logreg, f1_p_logreg)))
"""
data = get_user_data(df,39)
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

clf = GaussianProcessClassifier(RBF(1.0))
modelgaussian = create_model(clf)
clf = LogisticRegression(solver='liblinear', max_iter=400, C=1)
model = create_model(clf)
clf = MultinomialNB()
modelnaive = create_model(clf)
clf = SGDClassifier(average=True, max_iter=100)
modelsgd = create_model(clf)
f1_p_logreg = per_user_classification(df, model, False)
f1_p_sgd = per_user_classification(df, modelsgd, False)
f1_p_naive = per_user_classification(df, modelnaive, False)
f1_p_gaussian = per_user_classification(df, modelgaussian, False)

show_metric('Model F1-score for personal models ',
            'F1-score',
            ['sgd', 'logreg','bayes'],
            [f1_p_sgd, f1_p_logreg, f1_p_naive])

np.mean(f1_p_sgd)
np.mean(f1_p_logreg)
np.mean(f1_p_naive)
"""

