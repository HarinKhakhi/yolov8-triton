import pandas as pd 
from main import LOG_FILE, LOG_FIELDS
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(LOG_FILE, header=None, names=LOG_FIELDS)

ax = sns.violinplot(data=data, x='model_name', y='total_time')
plt.show()