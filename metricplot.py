import pandas as pd
from matplotlib import pyplot


df = pd.read_csv("df_metric_yearly_results.csv", usecols=["model","year","logloss","roc auc score","f1_score","MSE","accuracy","precision","recall",
                                                                  "pr auc score"])

# plot the precision-recall curves
df = df.loc[df['model'] == 'LightGBM Test set']
#df = df.loc[df['model'] == 'CatBoost Test set']

x = df['year']
y = df['f1_score']
pyplot.plot(x, y, marker='.',label = 'LightGBM')
#pyplot.plot(x, y, marker='.',label = 'CatBoost')

# axis labels
pyplot.xlabel('year')
pyplot.ylabel('f1 score')

# show the legend
pyplot.legend()
# show the plot
pyplot.show()

