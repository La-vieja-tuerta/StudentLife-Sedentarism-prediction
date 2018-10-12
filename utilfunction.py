from collections import Counter
import pandas as pd

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
    features = [col for col in df.columns if 'isSedentary' != col]
    return df[features].copy(), df['isSedentary'].copy()


def get_X_y_classification(df):
    df['sclass'] = ''
    df.loc[df['isSedentary'] > 0.9999, 'sclass'] = 0  # 'very sedentary'
    df.loc[df['isSedentary'].between(0.9052, 0.9999), 'sclass'] = 1  # 'sedentary'
    df.loc[df['isSedentary'] < 0.9052, 'sclass'] = 2  # 'less sedentary'
    features = [col for col in df.columns if 'sclass' != col]
    return df[features].copy(), df['sclass'].copy()
