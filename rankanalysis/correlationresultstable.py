import matplotlib.pyplot as plt
from prettytable import PrettyTable
import numpy as np
import pandas as pd


#######################################################################################################################
""""correlation table"""
#make a table where you have pcs uci corrleation for 2016/17/18..
#but also for several ranking: top 10,top20,top30,top40
myTable = PrettyTable(["Year", "Top 10", "Top 20", "Top 30", "Top 40"])
# Add rows
myTable.add_row(["2016", "0.48", "0.45", "0.37", "0.37"])
myTable.add_row(["2017", "0.57", "0.56", "0.46", "0.43"])
myTable.add_row(["2018", "0.48", "0.38", "0.33", "0.28"])
myTable.add_row(["2019", "0.61", "0.51", "0.41", "0.37"])
myTable.add_row(["2020", "0.39", "0.29", "0.24", "0.19"])
myTable.add_row(["2021", "0.46", "0.35", "0.28", "0.23"])
print(myTable)

#######################################################################################################################

"""correlation curve"""
topriders = [10,20,30,40]
correlation16 = [0.48,0.45,0.37,0.37]
correlation17 = [0.57,0.56,0.46,0.43]
correlation18 = [0.48,0.38,0.33,0.28]
correlation19 = [0.61,0.51,0.41,0.37]
correlation20 = [0.39,0.29,0.24,0.19]
correlation21 = [0.46,0.35,0.28,0.23]

plt.plot(topriders, correlation16)
plt.plot(topriders, correlation17)
plt.plot(topriders, correlation18)
plt.plot(topriders, correlation19)
plt.plot(topriders, correlation20)
plt.plot(topriders, correlation21)

#plt.title("Spearman Correlation curve between PCS and UCI for top 10/20/30/40 riders from 2016-2021")
plt.xlabel("Top Riders")
plt.ylabel("Spearman Correlation")
plt.legend(["Year 2016", "Year 2017", "Year 2018", "Year 2019","Year 2020","Year 2021"])
plt.show()






