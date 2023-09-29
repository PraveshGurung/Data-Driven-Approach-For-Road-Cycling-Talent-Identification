import pandas as pd
import glob
import matplotlib.pyplot as plt

def top100riderstocsv():
    path = "scrapped_data/top_riders_year/*.csv"

    col_list = ["rider_name","rider_slug","rider_url","team","pcs_points","team_slug","team_url","rank"]
    df_top_riders_all = pd.DataFrame([],columns=["rider_slug","rank","pcs_points"])

    for fname in glob.glob(path):
        df = pd.read_csv(fname, usecols=col_list)
        for index, row in df.iterrows():
            if row["rank"] <= 100:
                df_row = {"rider_slug": row["rider_slug"], "rank": row["rank"], "pcs_points":row["pcs_points"]}
                df_top_riders_all = df_top_riders_all.append(df_row, ignore_index=True)

    #to csv
    all_data = df_top_riders_all.to_csv("df_top_riders_all.csv",index=False)

def plot_numofriders_per_careerlength_historgram():
    #career year 1,2,3,4,5
    #number of riders
    #histogram
    # read_csv
    df_all = pd.read_csv("df_all.csv", usecols=["rider_name","year","Number_of_participated_one_day_races_current_year","Number_of_participated_multi_day_races_current_year",
                                      "win_ratio_all_time","top3_ratio_all_time","top5_ratio_all_time","top10_ratio_all_time",
                                      "stage_win_ratio_all_time","stage_top3_ratio_all_time","stage_top5_ratio_all_time","stage_top10_ratio_all_time",
                                      "win_ratio","top3_ratio","top5_ratio","top10_ratio","stage_win_ratio","stage_top3_ratio",
                                      "stage_top5_ratio","stage_top10_ratio","pcs_rank_previous_year", "pcs_rank_current_year",
                                      "pcs_points_previous_year","pcs_points_current_year",
                                      "rank_avg_last3years","pcs_points_avg_last3years",
                                      "age","career year","win_ratio_slope",
                                      "stage_win_ratio_slope","top3_ratio_slope","top5_ratio_slope","top10_ratio_slope",
                                      "stage_top3_ratio_slope","stage_top5_ratio_slope","stage_top10_ratio_slope",
                                      "win_ratio_last3years","top3_ratio_last3years","top5_ratio_last3years","top10_ratio_last3years",
                                      "stage_win_ratio_last3years","stage_top3_ratio_last3years","stage_top5_ratio_last3years","stage_top10_ratio_last3years",
                                      "Number_of_participated_one_day_races_all_time","Number_of_participated_multi_day_races_all_time",
                                      "Number_of_participated_one_day_races_last3years","Number_of_participated_multi_day_races_last3years",
                                      "rank_stdev_last3years","pcs_points_stdev_last3years",
                                      "win_ratio_stdev_last3years","top3_ratio_stdev_last3years",
                                      "top5_ratio_stdev_last3years","top10_ratio_stdev_last3years",
                                      "stage_win_ratio_stdev_last3years","stage_top3_ratio_stdev_last3years",
                                      "stage_top5_ratio_stdev_last3years","stage_top10_ratio_stdev_last3years",
                                      "result_tour-de-france","result_giro-d-italia","result_vuelta-a-espana","result_tour-down-under",
                                      "result_paris-nice","result_tirreno-adriatico","result_volta-a-catalunya","result_itzulia-basque-country",
                                      "result_tour-de-romandie","result_dauphine","result_tour-de-suisse","result_tour-de-pologne","result_benelux_tour",
                                      "result_milano-sanremo","result_ronde-van-vlaanderen","result_paris-roubaix","result_liege-bastogne-liege",
                                      "result_il-lombardia","result_strade-bianche","result_great-ocean-race","result_e3-harelbeke","result_gent-wevelgem",
                                      "result_la-fleche-wallone","result_amstel-gold-race","result_san-sebastian","result_bretagne-classic",
                                      "result_cyclassics-hamburg","result_gp-quebec","result_gp-montreal","in_top20_next_year"])

    min = df_all['career year'].min()
    max = df_all['career year'].max()

    careeryearlist = []
    numberofriders = []



    for i in range(df_all['career year'].min(),df_all['career year'].max()+1):
        career = df_all.loc[df_all['career year'] == i]
        careeryearlist.append(i)
        numberofriders.append(len(career.index))

    plt.title("Number of riders by career year")
    plt.bar(careeryearlist,numberofriders,align='center',color='darkorange') # A bar chart
    plt.xticks(careeryearlist,careeryearlist)
    plt.xlabel('Career year')
    plt.ylabel('Number of riders')
    plt.show()

import re

def replace_text_inside_brackets(input_string):
    pattern = r"\([^)]*\)"
    return re.sub(pattern, "$", input_string)



if __name__ == "__main__":
    top100riderstocsv()
    plot_numofriders_per_careerlength_historgram()
