#compare point distibution pcs uci:
######################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from scipy import stats
import seaborn as sbn

new_col_list=["rank1","rank2","rank3", "rank4","rank5","rank6","rank7","rank8","rank9","rank10"]

path = "compareyearsdata/pcs/rank*.csv"
df_pcs = pd.DataFrame([],columns=new_col_list)
x = []
for fname in glob.glob(path):
    df_temp = pd.read_csv(fname, usecols=["rank","count"])
    file_name = fname.split("pcs\\", 1)[1]
    file_name = file_name.split(".", 1)[0]
    df_pcs[file_name] = df_temp["count"]

    x += df_temp["count"].tolist()



path = "compareyearsdata/uci/rank*.csv"
df_uci = pd.DataFrame([],columns=new_col_list)
y = []
for fname in glob.glob(path):
    df_temp = pd.read_csv(fname, usecols=["rank","count"])
    file_name = fname.split("uci\\", 1)[1]
    file_name = file_name.split(".", 1)[0]
    df_uci[file_name] = df_temp["count"]
    y += df_temp["count"].tolist()



for rank in new_col_list:
    rho,pvalue = stats.spearmanr(df_pcs[rank],df_uci[rank])
    print(rank+" correlation : rho= %f , p-value= %f" % (rho,pvalue))

df = pd.DataFrame({'pcs': x,
                   'uci': y})
corr = df.corr(method='spearman')
print(corr)















