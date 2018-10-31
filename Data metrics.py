from utilfunction import *
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_pickle('sedentarismwithoutdummies')
df.drop(['audiomajor', 'hourofday'], axis=1, inplace=True)
df = makeDummies(df)
df = METcalculation(df)
df = makeSedentaryClasses(df)


plt.close()
a = df.groupby(df.index.get_level_values(0))['sclass'].apply(lambda x : np.sum(x==0)).values
b = df.groupby(df.index.get_level_values(0))['sclass'].apply(lambda x : np.sum(x==1)).values
c = df.groupby(df.index.get_level_values(0))['sclass'].count().values
a = a/c
b = b/c
users = np.arange(1,50)
plt.scatter(users, a)
plt.scatter(users, b)
plt.grid(True)
plt.xticks(users, users, rotation='vertical')
plt.show()

