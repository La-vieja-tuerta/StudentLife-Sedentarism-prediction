#chequear bien usuario 52

from utilfunction import *
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_pickle('sedentarismwithoutdummies')
df['met'] = METcalculation(df)
df['met'] = df['met'].astype('float')

plt.close()

dfresult = pd.DataFrame(df.groupby(['dayofweek','hourofday'], as_index=False)['met'].mean())
dfresult = dfresult.pivot(index='dayofweek', values='met', columns='hourofday')
sns.heatmap(dfresult, cmap='RdBu_r', center=1.5)
plt.title('Average MET')
plt.ylabel('Part of day')
plt.xlabel('Day of week')
plt.show()

plt.close()
dfresult = pd.DataFrame(df.groupby(['dayofweek','hourofday'])['met'].std())
dfresult.reset_index(inplace=True)
dfresult = dfresult.pivot(index='dayofweek', values='met', columns='hourofday')
sns.heatmap(dfresult, cmap='RdBu_r')
plt.title('Standard Deviation MET')
plt.ylabel('Part of day')
plt.xlabel('Day of week')
plt.show()

for u in df.index.get_level_values(0).drop_duplicates():
    plt.close()
    dfuser = get_user_data(df,u)
    userdata = pd.DataFrame(dfuser.groupby(['dayofweek','hourofday'], as_index=False)['met'].mean())
    userdata = userdata.pivot(index='dayofweek', values='met', columns='hourofday')
    sns.heatmap(userdata, cmap='RdBu_r', center=1.5)
    user = 'Average activity of user {0}'.format(u)
    plt.title(user)
    plt.ylabel('Part of day')
    plt.xlabel('Day of week')
    plt.show()

    plt.close()
    userdata = pd.DataFrame(dfuser.groupby(['dayofweek', 'hourofday'])['met'].std())
    userdata.reset_index(inplace=True)
    userdata = userdata.pivot(index='dayofweek', values='met', columns='hourofday')
    sns.heatmap(userdata)

    plt.title('Standard Deviation MET of user {0}'.format(u))
    plt.ylabel('Part of day')
    plt.xlabel('Day of week')
    plt.show()