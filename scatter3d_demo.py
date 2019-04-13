from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

stats = pd.read_pickle('stats.pkl')


X=stats['class']
Y=stats['std']
Z=stats['f1']

plt.close()
sns.lmplot(x='std',y='f1', data=stats, fit_reg=False)
plt.show()

c = np.ones(48)

