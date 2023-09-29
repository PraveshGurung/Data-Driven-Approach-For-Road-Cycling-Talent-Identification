import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#######################################################################################################################

"""normalized point distribution curve for one day and multi day race between pcs and uci for specific popular races"""
#tour de france
ucinew = [1000,800,675,575,475,400,325,275,225,175,150,125,105,85,75,70,65,60,55,50,40,40,40,40,40,30,30,30,30,30,25,25,25,25,25,25,25,25,25,25,20,20,20,20,20,20,20,20,20,20,15,15,15,15,15,10,10,10,10,10]
uciold = [200,150,120,110,100,90,80,70,60,50,40,30,24,20,16,12,10,8,6,4]
pcs = [500,380,340,300,280,260,240,220,210,200,190,180,170,160,150,140,130,120,110,100,95,90,85,80,75,70,65,60,55,50,45,40,35,30,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25]
"""
#giro d'italia
ucinew = [850,680,575,460,380,320,260,220,180,140,120,100,84,68,60,56,52,48,44,40,32,32,32,32,32,24,24,24,24,24,20,20,20,20,20,20,20,20,20,20,16,16,16,16,16,16,16,16,16,16,12,12,12,12,12,8,8,8,8,8]
uciold = [170,130,100,90,80,70,60,52,44,38,32,26,22,18,14,10,8,6,4,2]
pcs = [400,290,240,220,200,190,180,170,160,150,140,130,120,110,100,90,85,80,75,70,65,60,55,50,45,40,35,30,25,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
"""
"""
#tour de flanders
ucinew = [500,400,325,275,225,175,150,125,100,85,70,60,50,40,35,30,30,30,30,30,20,20,20,20,20,20,20,20,20,20,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,5,5,5,5,5,3,3,3,3,3]
uciold = [100,80,70,60,50,40,30,20,10,4]
pcs = [275,200,150,120,100,90,80,70,60,50,46,42,38,34,30,28,26,24,22,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
print(len(ucinew),len(uciold),len(pcs))
"""

#Normalization
ucinewmin, ucinewmax = min(ucinew), max(ucinew)
for i, val in enumerate(ucinew):
    ucinew[i] = (val-ucinewmin) / (ucinewmax-ucinewmin)

ucioldmin, ucioldmax = min(uciold), max(uciold)
for i, val in enumerate(uciold):
    uciold[i] = (val-ucioldmin) / (ucioldmax-ucioldmin)

pcsmin, pcsmax = min(pcs), max(pcs)
for i, val in enumerate(pcs):
    pcs[i] = (val-pcsmin) / (pcsmax-pcsmin)

rank = []
for i in range(60):
    rank.append(i+1)
rank2 = []
for i in range(20): #10 for flanders,20 for france/italia
    rank2.append(i+1)

plt.plot(rank, ucinew)
plt.plot(rank2, uciold)
plt.plot(rank, pcs)
#plt.title("Point distribution for Ronde van Vlaanderen")
#plt.title("Point distribution for Giro d'Italia")
plt.title("Point distribution for Tour de France")

plt.xlabel("rank")
plt.ylabel("points (normalized)")
plt.legend(["UCI New", "UCI Old", "PCS"])
plt.show()

#######################################################################################################################
"""normalized point distribution curve and normalized delta graph top 100/500 year 2016-2019 accumulated point"""

col_list = ["rider_name","year","pcs_points"]
df16pcs = pd.read_csv("../rank_data/top500/pcs/2016top_pcs.csv", usecols=col_list)
df17pcs = pd.read_csv("../rank_data/top500/pcs/2017top_pcs.csv", usecols=col_list)
df18pcs = pd.read_csv("../rank_data/top500/pcs/2018top_pcs.csv", usecols=col_list)
df19pcs = pd.read_csv("../rank_data/top500/pcs/2019top_pcs.csv", usecols=col_list)
pcs = (df16pcs[["pcs_points"]] + df17pcs[["pcs_points"]] + df18pcs[["pcs_points"]] + df19pcs[["pcs_points"]])

col_list2 = ["rider_name","year","uci_points"]
df16uci = pd.read_csv("../rank_data/top500/uci/2016top_uci.csv", usecols=col_list2)
df17uci = pd.read_csv("../rank_data/top500/uci/2017top_uci.csv", usecols=col_list2)
df18uci = pd.read_csv("../rank_data/top500/uci/2018top_uci.csv", usecols=col_list2)
df19uci = pd.read_csv("../rank_data/top500/uci/2019top_uci.csv", usecols=col_list2)
#uci = (df16uci[["uci_points"]] + df17uci[["uci_points"]] + df18uci[["uci_points"]] + df19uci[["uci_points"]])


pcs_points16 = df16pcs[["pcs_points"]]
pcs_points16 = pcs_points16["pcs_points"].tolist()
pcs_points17 = df17pcs[["pcs_points"]]
pcs_points17 = pcs_points17["pcs_points"].tolist()
pcs_points18 = df18pcs[["pcs_points"]]
pcs_points18 = pcs_points18["pcs_points"].tolist()
pcs_points19 = df19pcs[["pcs_points"]]
pcs_points19 = pcs_points19["pcs_points"].tolist()
uci_points16 = df16uci[["uci_points"]]
uci_points16 = uci_points16["uci_points"].tolist()
uci_points17 = df17uci[["uci_points"]]
uci_points17 = uci_points17["uci_points"].tolist()
uci_points18 = df18uci[["uci_points"]]
uci_points18 = uci_points18["uci_points"].tolist()
uci_points19 = df19uci[["uci_points"]]
uci_points19 = uci_points19["uci_points"].tolist()

#uci_points = uci["uci_points"].tolist()
#pcs_points = pcs["pcs_points"].tolist()

#normalization
ucimin16, ucimax16 = min(uci_points16), max(uci_points16)
for i, val in enumerate(uci_points16):
    uci_points16[i] = (val-ucimin16) / (ucimax16-ucimin16)
pcsmin16, pcsmax16 = min(pcs_points16), max(pcs_points16)
for i, val in enumerate(pcs_points16):
    pcs_points16[i] = (val-pcsmin16) / (pcsmax16-pcsmin16)


ucimin17, ucimax17 = min(uci_points17), max(uci_points17)
for i, val in enumerate(uci_points17):
    uci_points17[i] = (val-ucimin17) / (ucimax17-ucimin17)
pcsmin17, pcsmax17 = min(pcs_points17), max(pcs_points17)
for i, val in enumerate(pcs_points17):
    pcs_points17[i] = (val-pcsmin17) / (pcsmax17-pcsmin17)

ucimin18, ucimax18 = min(uci_points18), max(uci_points18)
for i, val in enumerate(uci_points18):
    uci_points18[i] = (val-ucimin18) / (ucimax18-ucimin18)
pcsmin18, pcsmax18 = min(pcs_points18), max(pcs_points18)
for i, val in enumerate(pcs_points18):
    pcs_points18[i] = (val-pcsmin18) / (pcsmax18-pcsmin18)

ucimin19, ucimax19 = min(uci_points19), max(uci_points19)
for i, val in enumerate(uci_points19):
    uci_points19[i] = (val-ucimin19) / (ucimax19-ucimin19)
pcsmin19, pcsmax19 = min(pcs_points19), max(pcs_points19)
for i, val in enumerate(pcs_points19):
    pcs_points19[i] = (val-pcsmin19) / (pcsmax19-pcsmin19)

difference16 = abs(np.array(pcs_points16)-np.array(uci_points16))
difference17 = abs(np.array(pcs_points17)-np.array(uci_points17))
difference18 = abs(np.array(pcs_points18)-np.array(uci_points18))
difference19 = abs(np.array(pcs_points19)-np.array(uci_points19))

rank = []
for i in range(500):
    rank.append(i+1)

"""normalized delta graph"""
#take median
difference = np.array([difference16,difference17,difference18,difference19])
median_delta = np.median(difference, axis=0)
#plt.plot(rank, median_delta)
plt.plot(rank[:100], median_delta[:100])
plt.xlabel("rank")
plt.ylabel("delta")
plt.show()

"""normalized point distribution curve"""
#take median
pcs_points = np.array([pcs_points16,pcs_points17,pcs_points18,pcs_points19])
median_pcs_points = np.median(pcs_points, axis=0)
uci_points = np.array([uci_points16,uci_points17,uci_points18,uci_points19])
median_uci_points = np.median(uci_points, axis=0)

#plt.plot(rank, median_uci_points)
#plt.plot(rank, median_pcs_points)
plt.plot(rank[:100], median_uci_points[:100])
plt.plot(rank[:100], median_pcs_points[:100])
plt.xlabel("rank")
plt.ylabel("points (normalized)")
plt.legend(["UCI","PCS"])
plt.show()