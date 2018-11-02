#chequear bien usuario 52
# probar sacando la cantidad de logs de cada actividad
from utilfunction import *
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_pickle('sedentarismwithoutdummies')
df = METcalculation(df)
df['slevel'] = df['slevel'].astype('float')

plt.close()

dfresult = pd.DataFrame(df.groupby(['dayofweek','hourofday'], as_index=False)['slevel'].mean())
dfresult = dfresult.pivot(index='dayofweek', values='slevel', columns='hourofday')
sns.heatmap(dfresult, cmap='RdBu_r', center=1.5)
plt.title('Average MET')
plt.ylabel('Part of day')
plt.xlabel('Day of week')
plt.show()

plt.close()
dfresult = pd.DataFrame(df.groupby(['dayofweek','hourofday'])['slevel'].std())
dfresult.reset_index(inplace=True)
dfresult = dfresult.pivot(index='dayofweek', values='slevel', columns='hourofday')
sns.heatmap(dfresult, cmap='RdBu_r')
plt.title('Standard Deviation MET')
plt.ylabel('Part of day')
plt.xlabel('Day of week')
plt.show()

for u in df.index.get_level_values(0).drop_duplicates():
    plt.close()
    dfuser = get_user_data(df,u)
    userdata = pd.DataFrame(dfuser.groupby(['dayofweek','hourofday'], as_index=False)['slevel'].mean())
    userdata = userdata.pivot(index='dayofweek', values='slevel', columns='hourofday')
    sns.heatmap(userdata, cmap='RdBu_r', center=1.5)
    user = 'Average activity of user {0}'.format(u)
    plt.title(user)
    plt.ylabel('Part of day')
    plt.xlabel('Day of week')
    plt.show()

    plt.close()
    userdata = pd.DataFrame(dfuser.groupby(['dayofweek', 'hourofday'])['slevel'].std())
    userdata.reset_index(inplace=True)
    userdata = userdata.pivot(index='dayofweek', values='slevel', columns='hourofday')
    sns.heatmap(userdata)

    plt.title('Standard Deviation MET of user {0}'.format(u))
    plt.ylabel('Part of day')
    plt.xlabel('Day of week')
    plt.show()