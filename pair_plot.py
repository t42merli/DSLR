import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('dataset_train.csv', index_col="Index")
data = data.dropna()

sns.pairplot(data, hue="Hogwarts House", vars=[ 'Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes',
                                               'History of Magic', 'Transfiguration', 'Potions', 'Charms', 'Flying'], markers=".", height=2)
plt.legend()
plt.show()
