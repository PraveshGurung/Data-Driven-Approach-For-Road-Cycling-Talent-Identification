import pandas as pd


col_list2 = ["rider_slug_year","year","rank","rider_slug","rider_name","rider_url","team_slug","points","last_update","rider_slug_year.1",
            "rider_slug_year.1.1","rider_slug_year"]
df = pd.read_csv("dataset/top_metadata.csv", usecols=col_list2)
pcs = df[["rider_name","year","rank","points"]]
meta_pcs = pcs.to_csv("rank_data/meta_pcs.csv",index=False)


col_list = ["race_id","race_slug","stage_slug","date","year","race_type","class","race_name",
            "race_url","rank","distance","pcs_points","uci_points","time_abs","time_rel","has_no_results"]

import glob
path = "dataset/riders/*.csv"
df_all = pd.DataFrame([],columns=["rider_name","year","pcs_points", "uci_points"])
for fname in glob.glob(path):
    df = pd.read_csv(fname, usecols=col_list)
    df = df[["year", "pcs_points", "uci_points"]]
    df = df.groupby(df["year"]).sum().reset_index()
    rider_name = fname.split("riders\\", 1)[1]
    rider_name = rider_name.split(".", 1)[0]
    rider_name = rider_name.replace("-"," ")
    df['rider_name'] = rider_name
    df_all = df_all.append(df)

#filter out years with less than 500 riders
v = df_all.year.value_counts()
df_all = df_all[df_all.year.isin(v.index[v.ge(500)])]

#uci ranking
df_uci = df_all[["rider_name","year","uci_points"]]
df_uci = df_uci.groupby("year").apply(lambda x: x.nlargest(500,"uci_points")) #get top 500 riders from each year
uci_csv = df_uci.to_csv("rank_data/uci.csv",index=False)

#pcs ranking
df_pcs = df_all[["rider_name","year","pcs_points"]]
df_pcs = df_pcs.groupby("year").apply(lambda x: x.nlargest(500,"pcs_points"))
pcs_csv = df_pcs.to_csv("rank_data/pcs.csv",index=False)

#years = [2010,2011,2012,2013,2014,2015]#2010-2015
years = [2016,2017,2018,2019,2020,2021] #WorldRanking introduced in 2016, can see drastic increase in points


for year in years:
    top_uci = df_uci.loc[df_uci["year"] == year]
    top_uci_csv = top_uci.to_csv("rank_data/top500/uci/"+str(year)+"top_uci.csv",index=False)
    a = pd.read_csv("rank_data/top500/uci/"+str(year)+"top_uci.csv")
    a.index += 1 #index start from 1 for ranking purposes
    a.to_html("rank_data/top500/uci/"+str(year)+"top_uci.html")

    top_pcs = df_pcs.loc[df_pcs["year"] == year]
    top_pcs_csv = top_pcs.to_csv("rank_data/top500/pcs/"+str(year)+"top_pcs.csv", index=False)
    a = pd.read_csv("rank_data/top500/pcs/"+str(year)+"top_pcs.csv")
    a.index += 1
    a.to_html("rank_data/top500/pcs/"+str(year)+"top_pcs.html")


#WorldRanking introduced in 2016
#WorldTour changed to be same as WorldRanking in 2017
#Worldranking: World Ranking includes points gained from any event in the UCI road calendar.
#This comprises: UCI WorldTour, UCI Continental Circuits, UCI World Championships, National and Continental Championships,
# Olympic Games and Continental Games.
#WorldTour : The WorldTour ranking only tallies points from races in the WorldTour calendar



