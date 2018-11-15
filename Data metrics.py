from utilfunction import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib.dates as dates

df = pd.read_pickle('sedentarismdata.pkl')
df = delete_user(df,52)
df = METcalculation(df)
df = makeSedentaryClasses(df)
df = makeDummies(df)
df = shift_hours(df,1, 'classification')
df.drop(['hourofday'],
        axis=1, inplace=True)


plt.close()
a = df.groupby(df.index.get_level_values(0))['sclass'].apply(lambda x : np.sum(x==0)).values
b = df.groupby(df.index.get_level_values(0))['sclass'].apply(lambda x : np.sum(x==1)).values
c = df.groupby(df.index.get_level_values(0))['sclass'].count().values
a = a/c
b = b/c
users = np.arange(1,49)
plt.scatter(users, a)
plt.scatter(users, b)
plt.grid(True)
plt.xticks(users, users, rotation='vertical')
plt.show()



xlabels = dates.date2num(pd.date_range("2013/4/08 00:00:00",periods=167, freq='h'))

user = get_user_data(df,8)
user = user.loc[(user.index.get_level_values(1)>"2013/4/08 00:00:00") &
                (user.index.get_level_values(1)<"2013/4/15 00:00")]
s = user['runningLevel'].values
w = user['walkingLevel'].values + user['runningLevel'].values
r = 1 - user['stationaryLevel'].values
x = np.arange(0,user.shape[0])

plt.close()
fig, ax = plt.subplots()
ax.plot(s, c='green', alpha=0.3)
ax.plot(r, c='red', alpha=0.3)
ax.fill_between(x,s,0, facecolor='orange', alpha=0.3)
ax.fill_between(x,w,s, facecolor='red', alpha=0.3)
ax.fill_between(x,r,1, facecolor='green', alpha=0.3)

ax.set_ylim([0.0,0.6])
plt.figure(figsize=(13,5))
plt.show()

#TODO agregar fechas al eje x