import pandas as pd
import matplotlib.pyplot as plt
import os
import errno
import glob
#pd.set_option("display.max_rows", None, "display.max_columns", None)



##########################
#look at the results for uci top10 riders from a certain year and see what it means to be in top 10: stats how often they finish 1st 2nd etc.
#so basically take their race ranks and type of race/race name and if they won any big races
#how did same rider rank in different systems: take a certain rider from top 10 and bring out at his stats: count all his place finishes
#just look at the results that i have outputted and compare
##########################

col_list = ["race_id","race_slug","stage_slug","date","year","race_type","class","race_name",
            "race_url","rank","distance","pcs_points","uci_points","time_abs","time_rel","has_no_results"]

#read 2015csv, get top 10 rider names, get rider csv from rider names and loop  through them and
#save the csv/html and plot


ranksys = "pcs"
col_list2 = ['rider_name','year',ranksys+'_points']


path = "rank_data/top500/"+ranksys+"/*.csv"
for fname in glob.glob(path):
    df = pd.read_csv(fname, usecols=col_list2) #year loop here
    df = df.head(40) #40
    df = df["rider_name"].tolist()
    year = fname.split(ranksys+"\\", 1)[1]
    year = year[0:4]

    path = "dataset/riders/*.csv"
    for fname in glob.glob(path):
        rider_name = fname.split("riders\\", 1)[1]
        rider_name = rider_name.split(".", 1)[0]
        rider_name = rider_name.replace("-", " ")
        if rider_name in df:
            rider_rank = df.index(rider_name) + 1
            df2 = pd.read_csv(fname, usecols=col_list)
            df2 = df2[["year", "rank", "race_type", "class"]]
            df2 = df2.loc[df2["year"] == int(year)]

            # count top 10 most frequent placings of the rider
            df2.drop(df2[df2['rank'] == "DNS"].index, inplace=True)
            df2.drop(df2[df2['rank'] == "DNF"].index, inplace=True)
            df2.drop(df2[df2['rank'] == "OTL"].index, inplace=True)
            df2.drop(df2[df2['rank'] == "DSQ"].index, inplace=True)
            df2.drop(df2[df2['rank'] == "DF"].index, inplace=True)
            rank = df2['rank'].value_counts().index.tolist()
            rank = list(map(int, rank))
            count = df2['rank'].value_counts().tolist()
            count = list(map(int, count))
            rider = pd.DataFrame(list(zip(rank, count)), columns=["rank", "count"])
            rider = rider.sort_values("rank")[:10]
            rider.plot.bar(x='rank', y='count')

            filename = "comparedata/"+ranksys+"/"+year+"topriders/"+str(rider_rank)+rider_name+"/"+rider_name+".csv"
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            plt.savefig("comparedata/"+ranksys+"/"+year+"topriders/"+str(rider_rank)+rider_name+"/"+rider_name+'.png')

            # count number of 1,2,3 finishes and top 10 and group by class
            for i in range(40): #40
                race_class = df2.loc[df2["rank"] == str(i+1)]
                race_class = race_class.groupby(["class", "rank"]).size()
                #race_class = race_class.groupby(["class"]).size()
                #race_class = pd.DataFrame({'class': race_class.index, 'count': race_class.values})
                race_class = race_class.to_csv("comparedata/"+ranksys+"/"+year+"topriders/"+str(rider_rank)+rider_name+"/"+str(rider_rank-1)+rider_name+str(i+1)+".csv", index=True)
                a = pd.read_csv("comparedata/"+ranksys+"/"+year+"topriders/"+str(rider_rank)+rider_name+"/"+str(rider_rank-1)+rider_name+str(i+1)+".csv")
                a.index += 1
                a.to_html("comparedata/"+ranksys+"/"+year+"topriders/"+str(rider_rank)+rider_name+"/"+rider_name+str(i+1)+".html")
                #a["count"] = pd.to_numeric(a["count"])
                #a = a.plot.bar(x='class', y='count')
                #a.set_ylim(0, 15)
                #plt.savefig("comparedata/"+ranksys+"/"+year+"topriders/"+str(rider_rank)+rider_name+"/"+rider_name+str(i+1)+".png")
                #break









