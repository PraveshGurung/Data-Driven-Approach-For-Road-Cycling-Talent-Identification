import pandas as pd

df_all = pd.read_csv("df_all.csv", usecols=["rider_name", "year", "Number_of_participated_one_day_races_current_year",
                                      "Number_of_participated_multi_day_races_current_year",
                                      "win_ratio_all_time", "top3_ratio_all_time", "top5_ratio_all_time",
                                      "top10_ratio_all_time","stage_win_ratio_all_time", "stage_top3_ratio_all_time",
                                      "stage_top5_ratio_all_time", "stage_top10_ratio_all_time",
                                      "win_ratio", "top3_ratio", "top5_ratio", "top10_ratio", "stage_win_ratio",
                                      "stage_top3_ratio","stage_top5_ratio", "stage_top10_ratio", "pcs_rank_previous_year",
                                      "pcs_rank_current_year","pcs_points_previous_year", "pcs_points_current_year",
                                      "rank_avg_last3years", "pcs_points_avg_last3years",
                                      "age", "career year","rider_bmi", "win_ratio_slope",
                                      "stage_win_ratio_slope", "top3_ratio_slope", "top5_ratio_slope",
                                      "top10_ratio_slope",
                                      "stage_top3_ratio_slope", "stage_top5_ratio_slope", "stage_top10_ratio_slope",
                                      "win_ratio_last3years", "top3_ratio_last3years", "top5_ratio_last3years",
                                      "top10_ratio_last3years",
                                      "stage_win_ratio_last3years", "stage_top3_ratio_last3years",
                                      "stage_top5_ratio_last3years", "stage_top10_ratio_last3years",
                                      "Number_of_participated_one_day_races_all_time",
                                      "Number_of_participated_multi_day_races_all_time",
                                      "Number_of_participated_one_day_races_last3years",
                                      "Number_of_participated_multi_day_races_last3years",
                                      "rank_stdev_last3years", "pcs_points_stdev_last3years",
                                      "win_ratio_stdev_last3years", "top3_ratio_stdev_last3years",
                                      "top5_ratio_stdev_last3years", "top10_ratio_stdev_last3years",
                                      "stage_win_ratio_stdev_last3years", "stage_top3_ratio_stdev_last3years",
                                      "stage_top5_ratio_stdev_last3years", "stage_top10_ratio_stdev_last3years",
                                      "result_tour-de-france", "result_giro-d-italia", "result_vuelta-a-espana",
                                      "result_tour-down-under",
                                      "result_paris-nice", "result_tirreno-adriatico", "result_volta-a-catalunya",
                                      "result_itzulia-basque-country",
                                      "result_tour-de-romandie", "result_dauphine", "result_tour-de-suisse",
                                      "result_tour-de-pologne", "result_benelux_tour",
                                      "result_milano-sanremo", "result_ronde-van-vlaanderen", "result_paris-roubaix",
                                      "result_liege-bastogne-liege",
                                      "result_il-lombardia", "result_strade-bianche", "result_great-ocean-race",
                                      "result_e3-harelbeke", "result_gent-wevelgem",
                                      "result_la-fleche-wallone", "result_amstel-gold-race", "result_san-sebastian",
                                      "result_bretagne-classic",
                                      "result_cyclassics-hamburg", "result_gp-quebec", "result_gp-montreal","actual_rank",
                                      "in_top20_next_year"])


df_curve = pd.DataFrame([],columns=["rider_name","year","Number_of_participated_one_day_races_current_year","Number_of_participated_multi_day_races_current_year",
                                  "win_ratio_all_time","top3_ratio_all_time","top5_ratio_all_time","top10_ratio_all_time",
                                  "stage_win_ratio_all_time","stage_top3_ratio_all_time","stage_top5_ratio_all_time","stage_top10_ratio_all_time",
                                  "win_ratio","top3_ratio","top5_ratio","top10_ratio","stage_win_ratio","stage_top3_ratio",
                                  "stage_top5_ratio","stage_top10_ratio","pcs_rank_previous_year", "pcs_rank_current_year",
                                  "pcs_points_previous_year","pcs_points_current_year",
                                  "rank_avg_last3years","pcs_points_avg_last3years",
                                  "age","career year","rider_bmi","win_ratio_slope",
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
                                  "result_cyclassics-hamburg","result_gp-quebec","result_gp-montreal","RankY1","RankY2","RankY3","RankY4","RankY5"])

#for each row in dfall,get year and rider name,
#look for rows in dfall with same rider name and for next 5 years,
#for those rows get the pcs rank current year
#make a new updated row and add it to dfcurve
#do all of this only for rows that have year =< 2016 <-need to change to 2017 later when i have scrapped year 2022 info aswell
for index, row in df_all.iterrows():
    rider_name = row["rider_name"]
    year = row["year"]
    if year <= 2017:
        year_one = df_all.loc[(df_all['rider_name'] == rider_name) & (df_all['year'] == year)]
        # if dataframe empty skip this row (the rider needs to compete for atleast 5 years in the future which was not the case)
        if year_one.empty:
            continue
        year_two = df_all.loc[(df_all['rider_name'] == rider_name) & (df_all['year'] == year + 1)]
        if year_two.empty:
            continue
        year_three = df_all.loc[(df_all['rider_name'] == rider_name) & (df_all['year'] == year + 2)]
        if year_three.empty:
            continue
        year_four = df_all.loc[(df_all['rider_name'] == rider_name) & (df_all['year'] == year + 3)]
        if year_four.empty:
            continue
        year_five = df_all.loc[(df_all['rider_name'] == rider_name) & (df_all['year'] == year + 4)]
        if year_five.empty:
            continue

        # from that row get the pcs rank current year
        rank_year_one = year_one["actual_rank"].values[0]
        rank_year_two = year_two["actual_rank"].values[0]
        rank_year_three = year_three["actual_rank"].values[0]
        rank_year_four = year_four["actual_rank"].values[0]
        rank_year_five = year_five["actual_rank"].values[0]

        #add all ranks to a list
        #fiveyearrankslist = [rank_year_one,rank_year_two,rank_year_three,rank_year_four,rank_year_five]

        #drop intop20nextyear from row
        df_row = df_all.loc[(df_all['rider_name'] == rider_name) & (df_all['year'] == year)]
        df_row = df_row.drop("in_top20_next_year",axis=1)

        #add new column to it
        #df_row["5_year_ranks"] = [fiveyearrankslist]
        df_row["RankY1"] = rank_year_one
        df_row["RankY2"] = rank_year_two
        df_row["RankY3"] = rank_year_three
        df_row["RankY4"] = rank_year_four
        df_row["RankY5"] = rank_year_five

        #add that row to dfcurve
        df_curve = df_curve.append(df_row, ignore_index=True)

#to csv
all_data = df_curve.to_csv("df_all_curve.csv",index=False)