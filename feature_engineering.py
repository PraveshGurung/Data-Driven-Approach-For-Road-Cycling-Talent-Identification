import numpy as np
import pandas as pd
import glob
from scipy.stats import linregress
from pcs_downloader.downloader import Downloader
from pathlib import Path
from writers.writer import Writer
from writers.writer_basic import WriterBasic
from datetime import date, datetime
import country_converter as coco
import statistics

# read_csv
df_top_riders_all = pd.read_csv("df_top_riders_all.csv", usecols=["rider_slug", "rank", "pcs_points"])
# --------------------------------------------------------------------------------------------------------------------------------------------------

# The path to the different data
data_path = Path('scrapped_data/')
races_path = data_path / 'races'
riders_path = data_path / 'riders'
race_metadata_path = data_path / 'race_metadata.csv'
rider_metadata_path = data_path / 'rider_metadata.csv'
top_rider_year_path = data_path / 'top_riders_year'

race_result_io = WriterBasic(races_path)
rider_result_io = WriterBasic(riders_path)
race_metadata = Writer(race_metadata_path, 'race_id')
rider_metadata = Writer(rider_metadata_path, 'rider_slug')
top_rider_year_io = WriterBasic(top_rider_year_path)
error_while_getting_start_list = []

downloader = Downloader(race_result_io, rider_result_io, race_metadata, rider_metadata, top_rider_year_io)


def calc_age(birthdate, target_date):
    return target_date.year - birthdate.year - ((target_date.month, target_date.day) < (birthdate.month, birthdate.day))


# feature engineering
col_list = ["race_id", "race_slug", "stage_slug", "date", "year", "race_type", "class", "race_name",
            "race_url", "rank", "distance", "pcs_points", "uci_points", "time_abs", "time_rel", "has_no_results"]

path = "scrapped_data/riders/*.csv"
df_all = pd.DataFrame([], columns=["rider_name", "year", "Number_of_participated_one_day_races_current_year",
                                   "Number_of_participated_multi_day_races_current_year",
                                   "win_ratio_all_time", "top3_ratio_all_time", "top5_ratio_all_time",
                                   "top10_ratio_all_time",
                                   "stage_win_ratio_all_time", "stage_top3_ratio_all_time", "stage_top5_ratio_all_time",
                                   "stage_top10_ratio_all_time",
                                   "win_ratio", "top3_ratio", "top5_ratio", "top10_ratio", "stage_win_ratio",
                                   "stage_top3_ratio",
                                   "stage_top5_ratio", "stage_top10_ratio", "pcs_rank_previous_year",
                                   "pcs_rank_current_year",
                                   "pcs_points_previous_year", "pcs_points_current_year",
                                   "rank_avg_last3years", "pcs_points_avg_last3years",
                                   "age", "career year", "rider_bmi", "win_ratio_slope",
                                   "stage_win_ratio_slope", "top3_ratio_slope", "top5_ratio_slope", "top10_ratio_slope",
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
                                   "result_cyclassics-hamburg", "result_gp-quebec", "result_gp-montreal", "actual_rank",
                                   "in_top20_next_year"])

# years = [2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
# years = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009] #atleast 5 years because of slope feature which looks 5 years back including currnet year
years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]

meta_col_list = ["rider_slug", "rider_name", "birthday", "nationality", "weight", "height", "place_of_birth",
                 "instagram", "strava", "facebook", "twitter", "pcs", "website",
                 "pcs_photo_url", "last_update"]
rider_meta_df = pd.read_csv("scrapped_data/rider_metadata.csv", usecols=meta_col_list)

for fname in glob.glob(path):  #
    df = pd.read_csv(fname, usecols=col_list)
    rider_slug = fname.split("riders\\", 1)[1]
    rider_slug = rider_slug.split(".", 1)[0]
    col_list2 = ["rider_name", "rider_slug", "rider_url", "team", "pcs_points", "team_slug", "team_url", "rank"]

    if years[-1] == 2021:  # here
        # years = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009]
        years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]
    loop = 2022 - years[-1]  # here
    for i in range(loop):
        label_df = pd.read_csv("scrapped_data/top_riders_year/" + str(years[-1] + 1) + ".csv", usecols=col_list2)
        current_df = pd.read_csv("scrapped_data/top_riders_year/" + str(years[-1]) + ".csv", usecols=col_list2)
        prev_df = pd.read_csv("scrapped_data/top_riders_year/" + str(years[-2]) + ".csv", usecols=col_list2)
        prev2_df = pd.read_csv("scrapped_data/top_riders_year/" + str(years[-3]) + ".csv", usecols=col_list2)

        label_riders = label_df['rider_slug'].tolist()
        current_riders = current_df['rider_slug'].tolist()
        prev_riders = prev_df['rider_slug'].tolist()
        prev2_riders = prev2_df['rider_slug'].tolist()

        # check if rider in top 100 in any of the years if yes, check if rider in current year (aka scrapped data = top500) if yes go further if no skip
        top100riders = df_top_riders_all['rider_slug'].tolist()
        if rider_slug in top100riders:
            if rider_slug in current_riders:
                year = years[-1]  # currentyear
                getfirstyear = df.loc[df['year'].isin(years)]
                firstyear = getfirstyear['year'].min()
                career_year = year - firstyear + 1
                if career_year >= 3:  # only take riders that have at least 3 years of experience
                    # downloader.rider_and_races(result['rider_slug'], target_race_year) #rider parse if outside scrapped data (top 500)
                    current = current_df.loc[current_df['rider_slug'] == rider_slug]
                    current_rank = current['rank'].values[0]

                    # if current_rank <= 100:
                    df2 = df[df['race_type'] == 'one_day'].loc[
                        df['year'].isin(years)]  # only select stage_slug = nan (non-stage entries), from given years

                    # na = not aggregated, thus by year
                    df2_na = df[df['race_type'] == 'one_day'].loc[
                        df['year'] == years[-1]]  # only select stage_slug = nan (non-stage entries), from given years
                    df2_na_prev = df[df['race_type'] == 'one_day'].loc[df['year'] == years[-2]]
                    df2_na_prev2 = df[df['race_type'] == 'one_day'].loc[df['year'] == years[-3]]

                    # last 3 years
                    df2_last3years = df[df['race_type'] == 'one_day'].loc[
                        df['year'].isin([years[-3], years[-2], years[-1]])]

                    number_races = len(df2.index)

                    try:
                        wins = df2['rank'].value_counts()['1']
                        win_ratio = wins / number_races
                    except KeyError:
                        win_ratio = 0

                    top3 = 0
                    for i in range(3):
                        try:
                            top3 += df2['rank'].value_counts()[str(i + 1)]
                        except KeyError:
                            pass
                    top3_ratio = 0
                    if number_races != 0:
                        top3_ratio = top3 / number_races

                    top5 = 0
                    for i in range(5):
                        try:
                            top5 += df2['rank'].value_counts()[str(i + 1)]
                        except KeyError:
                            pass
                    top5_ratio = 0
                    if number_races != 0:
                        top5_ratio = top5 / number_races

                    top10 = 0
                    for i in range(10):
                        try:
                            top10 += df2['rank'].value_counts()[str(i + 1)]
                        except KeyError:
                            pass
                    top10_ratio = 0
                    if number_races != 0:
                        top10_ratio = top10 / number_races

                    number_races_last3years = len(df2_last3years.index)

                    try:
                        wins_last3years = df2_last3years['rank'].value_counts()['1']
                        win_ratio_last3years = wins_last3years / number_races_last3years
                    except KeyError:
                        win_ratio_last3years = 0

                    top3_last3years = 0
                    for i in range(3):
                        try:
                            top3_last3years += df2_last3years['rank'].value_counts()[str(i + 1)]
                        except KeyError:
                            pass
                    top3_ratio_last3years = 0
                    if number_races_last3years != 0:
                        top3_ratio_last3years = top3_last3years / number_races_last3years

                    top5_last3years = 0
                    for i in range(5):
                        try:
                            top5_last3years += df2_last3years['rank'].value_counts()[str(i + 1)]
                        except KeyError:
                            pass
                    top5_ratio_last3years = 0
                    if number_races_last3years != 0:
                        top5_ratio_last3years = top5_last3years / number_races_last3years

                    top10_last3years = 0
                    for i in range(10):
                        try:
                            top10_last3years += df2_last3years['rank'].value_counts()[str(i + 1)]
                        except KeyError:
                            pass
                    top10_ratio_last3years = 0
                    if number_races_last3years != 0:
                        top10_ratio_last3years = top10_last3years / number_races_last3years

                    number_races_na = len(df2_na.index)
                    number_races_na_prev = len(df2_na_prev.index)
                    number_races_na_prev2 = len(df2_na_prev2.index)
                    win_ratio_na_prev2, win_ratio_na_prev, win_ratio_na = 0, 0, 0
                    top3_ratio_na_prev2, top3_ratio_na_prev, top3_ratio_na = 0, 0, 0
                    top5_ratio_na_prev2, top5_ratio_na_prev, top5_ratio_na = 0, 0, 0
                    top10_ratio_na_prev2, top10_ratio_na_prev, top10_ratio_na = 0, 0, 0
                    if df2_na_prev.empty or df2_na_prev2.empty:
                        win_ratio_slope = np.nan
                        top3_ratio_slope = np.nan
                        top5_ratio_slope = np.nan
                        top10_ratio_slope = np.nan
                    else:
                        try:
                            wins_na = df2_na['rank'].value_counts()['1']
                            win_ratio_na = wins_na / number_races_na
                        except KeyError:
                            win_ratio_na = 0

                        try:
                            wins_na_prev = df2_na_prev['rank'].value_counts()['1']
                            win_ratio_na_prev = wins_na_prev / number_races_na_prev
                        except KeyError:
                            win_ratio_na_prev = 0

                        try:
                            wins_na_prev2 = df2_na_prev2['rank'].value_counts()['1']
                            win_ratio_na_prev2 = wins_na_prev2 / number_races_na_prev2
                        except KeyError:
                            win_ratio_na_prev2 = 0

                        ratio, intercept, r_value, p_value, std_err = linregress([years[-3], years[-2], years[-1]],
                                                                                 [win_ratio_na_prev2, win_ratio_na_prev,
                                                                                  win_ratio_na])
                        win_ratio_slope = ratio

                        top3_na = 0
                        top3_na_prev = 0
                        top3_na_prev2 = 0
                        for i in range(3):
                            try:
                                top3_na += df2_na['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                            try:
                                top3_na_prev += df2_na_prev['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                            try:
                                top3_na_prev2 += df2_na_prev2['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                        if number_races_na != 0:
                            top3_ratio_na = top3_na / number_races_na
                        if number_races_na_prev != 0:
                            top3_ratio_na_prev = top3_na_prev / number_races_na_prev
                        if number_races_na_prev2 != 0:
                            top3_ratio_na_prev2 = top3_na_prev2 / number_races_na_prev2

                        ratio, intercept, r_value, p_value, std_err = linregress([years[-3], years[-2], years[-1]],
                                                                                 [top3_ratio_na_prev2,
                                                                                  top3_ratio_na_prev,
                                                                                  top3_ratio_na])
                        top3_ratio_slope = ratio

                        top5_na = 0
                        top5_na_prev = 0
                        top5_na_prev2 = 0
                        for i in range(5):
                            try:
                                top5_na += df2_na['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                            try:
                                top5_na_prev += df2_na_prev['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                            try:
                                top5_na_prev2 += df2_na_prev2['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                        if number_races_na != 0:
                            top5_ratio_na = top5_na / number_races_na
                        if number_races_na_prev != 0:
                            top5_ratio_na_prev = top5_na_prev / number_races_na_prev
                        if number_races_na_prev2 != 0:
                            top5_ratio_na_prev2 = top5_na_prev2 / number_races_na_prev2

                        ratio, intercept, r_value, p_value, std_err = linregress([years[-3], years[-2], years[-1]],
                                                                                 [top5_ratio_na_prev2,
                                                                                  top5_ratio_na_prev,
                                                                                  top5_ratio_na])
                        top5_ratio_slope = ratio

                        top10_na = 0
                        top10_na_prev = 0
                        top10_na_prev2 = 0
                        for i in range(10):
                            try:
                                top10_na += df2_na['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                            try:
                                top10_na_prev += df2_na_prev['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                            try:
                                top10_na_prev2 += df2_na_prev2['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                        if number_races_na != 0:
                            top10_ratio_na = top10_na / number_races_na
                        if number_races_na_prev != 0:
                            top10_ratio_na_prev = top10_na_prev / number_races_na_prev
                        if number_races_na_prev2 != 0:
                            top10_ratio_na_prev2 = top10_na_prev2 / number_races_na_prev2

                        ratio, intercept, r_value, p_value, std_err = linregress([years[-3], years[-2], years[-1]],
                                                                                 [top10_ratio_na_prev2,
                                                                                  top10_ratio_na_prev,
                                                                                  top10_ratio_na])
                        top10_ratio_slope = ratio

                    df3 = df[df['race_type'] != 'one_day'].loc[
                        df['year'].isin(years)]  # only select stage_slug is not nan (stage entries), from given years

                    # na = not aggregated, thus by year
                    df3_na = df[df['race_type'] != 'one_day'].loc[
                        df['year'] == years[-1]]  # only select stage_slug is not nan (stage entries), from given years
                    df3_na_prev = df[df['race_type'] != 'one_day'].loc[df['year'] == years[-2]]
                    df3_na_prev2 = df[df['race_type'] != 'one_day'].loc[df['year'] == years[-3]]

                    # last 3 years
                    df3_last3years = df[df['race_type'] != 'one_day'].loc[
                        df['year'].isin([years[-3], years[-2], years[-1]])]

                    number_stage_races = len(df3.index)

                    try:
                        stage_wins = df3['rank'].value_counts()['1']
                        stage_win_ratio = stage_wins / number_stage_races
                    except KeyError:
                        stage_win_ratio = 0

                    stage_top3 = 0
                    for i in range(3):
                        try:
                            stage_top3 += df3['rank'].value_counts()[str(i + 1)]
                        except KeyError:
                            pass
                    stage_top3_ratio = 0
                    if number_stage_races != 0:
                        stage_top3_ratio = stage_top3 / number_stage_races

                    stage_top5 = 0
                    for i in range(5):
                        try:
                            stage_top5 += df3['rank'].value_counts()[str(i + 1)]
                        except KeyError:
                            pass
                    stage_top5_ratio = 0
                    if number_stage_races != 0:
                        stage_top5_ratio = stage_top5 / number_stage_races

                    stage_top10 = 0
                    for i in range(10):
                        try:
                            stage_top10 += df3['rank'].value_counts()[str(i + 1)]
                        except KeyError:
                            pass
                    stage_top10_ratio = 0
                    if number_stage_races != 0:
                        stage_top10_ratio = stage_top10 / number_stage_races

                    number_stage_races_last3years = len(df3_last3years.index)

                    try:
                        stage_wins_last3years = df3_last3years['rank'].value_counts()['1']
                        stage_win_ratio_last3years = stage_wins_last3years / number_stage_races_last3years
                    except KeyError:
                        stage_win_ratio_last3years = 0

                    stage_top3_last3years = 0
                    for i in range(3):
                        try:
                            stage_top3_last3years += df3_last3years['rank'].value_counts()[str(i + 1)]
                        except KeyError:
                            pass
                    stage_top3_ratio_last3years = 0
                    if number_stage_races_last3years != 0:
                        stage_top3_ratio_last3years = stage_top3_last3years / number_stage_races_last3years

                    stage_top5_last3years = 0
                    for i in range(5):
                        try:
                            stage_top5_last3years += df3_last3years['rank'].value_counts()[str(i + 1)]
                        except KeyError:
                            pass
                    stage_top5_ratio_last3years = 0
                    if number_stage_races_last3years != 0:
                        stage_top5_ratio_last3years = stage_top5_last3years / number_stage_races_last3years

                    stage_top10_last3years = 0
                    for i in range(10):
                        try:
                            stage_top10_last3years += df3_last3years['rank'].value_counts()[str(i + 1)]
                        except KeyError:
                            pass
                    stage_top10_ratio_last3years = 0
                    if number_stage_races_last3years != 0:
                        stage_top10_ratio_last3years = stage_top10_last3years / number_stage_races_last3years

                    number_stage_races_na = len(df3_na.index)
                    number_stage_races_na_prev = len(df3_na_prev.index)
                    number_stage_races_na_prev2 = len(df3_na_prev2.index)

                    stage_win_ratio_na_prev2, stage_win_ratio_na_prev, stage_win_ratio_na = 0, 0, 0
                    stage_top3_ratio_na_prev2, stage_top3_ratio_na_prev, stage_top3_ratio_na = 0, 0, 0
                    stage_top5_ratio_na_prev2, stage_top5_ratio_na_prev, stage_top5_ratio_na = 0, 0, 0
                    stage_top10_ratio_na_prev2, stage_top10_ratio_na_prev, stage_top10_ratio_na = 0, 0, 0
                    if df3_na_prev.empty or df3_na_prev2.empty:
                        stage_win_ratio_slope = np.nan
                        stage_top3_ratio_slope = np.nan
                        stage_top5_ratio_slope = np.nan
                        stage_top10_ratio_slope = np.nan
                    else:
                        try:
                            stage_wins_na = df3_na['rank'].value_counts()['1']
                            stage_win_ratio_na = stage_wins_na / number_stage_races_na
                        except KeyError:
                            stage_win_ratio_na = 0

                        try:
                            stage_wins_na_prev = df3_na_prev['rank'].value_counts()['1']
                            stage_win_ratio_na_prev = stage_wins_na_prev / number_stage_races_na_prev
                        except KeyError:
                            stage_win_ratio_na_prev = 0

                        try:
                            stage_wins_na_prev2 = df3_na_prev2['rank'].value_counts()['1']
                            stage_win_ratio_na_prev2 = stage_wins_na_prev2 / number_stage_races_na_prev2
                        except KeyError:
                            stage_win_ratio_na_prev2 = 0

                        ratio, intercept, r_value, p_value, std_err = linregress([years[-3], years[-2], years[-1]],
                                                                                 [stage_win_ratio_na_prev2,
                                                                                  stage_win_ratio_na_prev,
                                                                                  stage_win_ratio_na])
                        stage_win_ratio_slope = ratio

                        stage_top3_na = 0
                        stage_top3_na_prev = 0
                        stage_top3_na_prev2 = 0
                        for i in range(3):
                            try:
                                stage_top3_na += df3_na['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                            try:
                                stage_top3_na_prev += df3_na_prev['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                            try:
                                stage_top3_na_prev2 += df3_na_prev2['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                        if number_stage_races_na != 0:
                            stage_top3_ratio_na = stage_top3_na / number_stage_races_na
                        if number_stage_races_na_prev != 0:
                            stage_top3_ratio_na_prev = stage_top3_na_prev / number_stage_races_na_prev
                        if number_stage_races_na_prev2 != 0:
                            stage_top3_ratio_na_prev2 = stage_top3_na_prev2 / number_stage_races_na_prev2

                        ratio, intercept, r_value, p_value, std_err = linregress([years[-3], years[-2], years[-1]],
                                                                                 [stage_top3_ratio_na_prev2,
                                                                                  stage_top3_ratio_na_prev,
                                                                                  stage_top3_ratio_na])
                        stage_top3_ratio_slope = ratio

                        stage_top5_na = 0
                        stage_top5_na_prev = 0
                        stage_top5_na_prev2 = 0
                        for i in range(5):
                            try:
                                stage_top5_na += df3_na['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                            try:
                                stage_top5_na_prev += df3_na_prev['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                            try:
                                stage_top5_na_prev2 += df3_na_prev2['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                        if number_stage_races_na != 0:
                            stage_top5_ratio_na = stage_top5_na / number_stage_races_na
                        if number_stage_races_na_prev != 0:
                            stage_top5_ratio_na_prev = stage_top5_na_prev / number_stage_races_na_prev
                        if number_stage_races_na_prev2 != 0:
                            stage_top5_ratio_na_prev2 = stage_top5_na_prev2 / number_stage_races_na_prev2

                        ratio, intercept, r_value, p_value, std_err = linregress([years[-3], years[-2], years[-1]],
                                                                                 [stage_top5_ratio_na_prev2,
                                                                                  stage_top5_ratio_na_prev,
                                                                                  stage_top5_ratio_na])
                        stage_top5_ratio_slope = ratio

                        stage_top10_na = 0
                        stage_top10_na_prev = 0
                        stage_top10_na_prev2 = 0
                        for i in range(10):
                            try:
                                stage_top10_na += df3_na['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                            try:
                                stage_top10_na_prev += df3_na_prev['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                            try:
                                stage_top10_na_prev2 += df3_na_prev2['rank'].value_counts()[str(i + 1)]
                            except KeyError:
                                pass
                        if number_stage_races_na != 0:
                            stage_top10_ratio_na = stage_top10_na / number_stage_races_na
                        if number_stage_races_na_prev != 0:
                            stage_top10_ratio_na_prev = stage_top10_na_prev / number_stage_races_na_prev
                        if number_stage_races_na_prev2 != 0:
                            stage_top10_ratio_na_prev2 = stage_top10_na_prev2 / number_stage_races_na_prev2

                        ratio, intercept, r_value, p_value, std_err = linregress([years[-3], years[-2], years[-1]],
                                                                                 [stage_top10_ratio_na_prev2,
                                                                                  stage_top10_ratio_na_prev,
                                                                                  stage_top10_ratio_na])
                        stage_top10_ratio_slope = ratio

                    tourdefrance = df.loc[df['race_id'] == "tour-de-france_" + str(year)]
                    result_tourdefrance = np.nan
                    if not tourdefrance.empty:
                        rank = tourdefrance['rank'].values[0].isdigit()
                        if rank:
                            result_tourdefrance = int(tourdefrance['rank'].values[0])

                    giroditalia = df.loc[df['race_id'] == "giro-d-italia_" + str(year)]
                    result_giroditalia = np.nan
                    if not giroditalia.empty:
                        rank = giroditalia['rank'].values[0].isdigit()
                        if rank:
                            result_giroditalia = int(giroditalia['rank'].values[0])

                    vueltaespana = df.loc[df['race_id'] == "vuelta-a-espana_" + str(year)]
                    result_vueltaespana = np.nan
                    if not vueltaespana.empty:
                        rank = vueltaespana['rank'].values[0].isdigit()
                        if rank:
                            result_vueltaespana = int(vueltaespana['rank'].values[0])

                    tourdownunder = df.loc[df['race_id'] == "tour-down-under_" + str(year)]
                    result_tourdownunder = np.nan
                    if not tourdownunder.empty:
                        rank = tourdownunder['rank'].values[0].isdigit()
                        if rank:
                            result_tourdownunder = int(tourdownunder['rank'].values[0])

                    parisnice = df.loc[df['race_id'] == "paris-nice_" + str(year)]
                    result_parisnice = np.nan
                    if not parisnice.empty:
                        rank = parisnice['rank'].values[0].isdigit()
                        if rank:
                            result_parisnice = int(parisnice['rank'].values[0])

                    tirrenoadriatico = df.loc[df['race_id'] == "tirreno-adriatico_" + str(year)]
                    result_tirrenoadriatico = np.nan
                    if not tirrenoadriatico.empty:
                        rank = tirrenoadriatico['rank'].values[0].isdigit()
                        if rank:
                            result_tirrenoadriatico = int(tirrenoadriatico['rank'].values[0])

                    voltaacatalunya = df.loc[df['race_id'] == "volta-a-catalunya_" + str(year)]
                    result_voltaacatalunya = np.nan
                    if not voltaacatalunya.empty:
                        rank = voltaacatalunya['rank'].values[0].isdigit()
                        if rank:
                            result_voltaacatalunya = int(voltaacatalunya['rank'].values[0])

                    itzuliabasquecountry = df.loc[df['race_id'] == "itzulia-basque-country_" + str(year)]
                    result_itzuliabasquecountry = np.nan
                    if not itzuliabasquecountry.empty:
                        rank = itzuliabasquecountry['rank'].values[0].isdigit()
                        if rank:
                            result_itzuliabasquecountry = int(itzuliabasquecountry['rank'].values[0])

                    tourderomandie = df.loc[df['race_id'] == "tour-de-romandie_" + str(year)]
                    result_tourderomandie = np.nan
                    if not tourderomandie.empty:
                        rank = tourderomandie['rank'].values[0].isdigit()
                        if rank:
                            result_tourderomandie = int(tourderomandie['rank'].values[0])

                    dauphine = df.loc[df['race_id'] == "dauphine_" + str(year)]
                    result_dauphine = np.nan
                    if not dauphine.empty:
                        rank = dauphine['rank'].values[0].isdigit()
                        if rank:
                            result_dauphine = int(dauphine['rank'].values[0])

                    tourdesuisse = df.loc[df['race_id'] == "tour-de-suisse_" + str(year)]
                    result_tourdesuisse = np.nan
                    if not tourdesuisse.empty:
                        rank = tourdesuisse['rank'].values[0].isdigit()
                        if rank:
                            result_tourdesuisse = int(tourdesuisse['rank'].values[0])

                    tourdepologne = df.loc[df['race_id'] == "tour-de-pologne_" + str(year)]
                    result_tourdepologne = np.nan
                    if not tourdepologne.empty:
                        rank = tourdepologne['rank'].values[0].isdigit()
                        if rank:
                            result_tourdepologne = int(tourdepologne['rank'].values[0])

                    beneluxtour = df.loc[df['race_id'] == "benelux-tour_" + str(year)]
                    result_beneluxtour = np.nan
                    if not beneluxtour.empty:
                        rank = beneluxtour['rank'].values[0].isdigit()
                        if rank:
                            result_beneluxtour = int(beneluxtour['rank'].values[0])

                    milanosanremo = df.loc[df['race_id'] == "milano-sanremo_" + str(year)]
                    result_milanosanremo = np.nan
                    if not milanosanremo.empty:
                        rank = milanosanremo['rank'].values[0].isdigit()
                        if rank:
                            result_milanosanremo = int(milanosanremo['rank'].values[0])

                    rondevanvlaanderen = df.loc[df['race_id'] == "ronde-van-vlaanderen_" + str(year)]
                    result_rondevanvlaanderen = np.nan
                    if not rondevanvlaanderen.empty:
                        rank = rondevanvlaanderen['rank'].values[0].isdigit()
                        if rank:
                            result_rondevanvlaanderen = int(rondevanvlaanderen['rank'].values[0])

                    parisroubaix = df.loc[df['race_id'] == "paris-roubaix_" + str(year)]
                    result_parisroubaix = np.nan
                    if not parisroubaix.empty:
                        rank = parisroubaix['rank'].values[0].isdigit()
                        if rank:
                            result_parisroubaix = int(parisroubaix['rank'].values[0])

                    liegebastogneliege = df.loc[df['race_id'] == "liege-bastogne-liege_" + str(year)]
                    result_liegebastogneliege = np.nan
                    if not liegebastogneliege.empty:
                        rank = liegebastogneliege['rank'].values[0].isdigit()
                        if rank:
                            result_liegebastogneliege = int(liegebastogneliege['rank'].values[0])

                    illombardia = df.loc[df['race_id'] == "il-lombardia_" + str(year)]
                    result_illombardia = np.nan
                    if not illombardia.empty:
                        rank = illombardia['rank'].values[0].isdigit()
                        if rank:
                            result_illombardia = int(illombardia['rank'].values[0])

                    stradebianche = df.loc[df['race_id'] == "strade-bianche_" + str(year)]
                    result_stradebianche = np.nan
                    if not stradebianche.empty:
                        rank = stradebianche['rank'].values[0].isdigit()
                        if rank:
                            result_stradebianche = int(stradebianche['rank'].values[0])

                    greatoceanrace = df.loc[df['race_id'] == "great-ocean-race_" + str(year)]
                    result_greatoceanrace = np.nan
                    if not greatoceanrace.empty:
                        rank = greatoceanrace['rank'].values[0].isdigit()
                        if rank:
                            result_greatoceanrace = int(greatoceanrace['rank'].values[0])

                    e3harelbeke = df.loc[df['race_id'] == "e3-harelbeke_" + str(year)]
                    result_e3harelbeke = np.nan
                    if not e3harelbeke.empty:
                        rank = e3harelbeke['rank'].values[0].isdigit()
                        if rank:
                            result_e3harelbeke = int(e3harelbeke['rank'].values[0])

                    gentwevelgem = df.loc[df['race_id'] == "gent-wevelgem_" + str(year)]
                    result_gentwevelgem = np.nan
                    if not gentwevelgem.empty:
                        rank = gentwevelgem['rank'].values[0].isdigit()
                        if rank:
                            result_gentwevelgem = int(gentwevelgem['rank'].values[0])

                    laflechewallone = df.loc[df['race_id'] == "la-fleche-wallone_" + str(year)]
                    result_laflechewallone = np.nan
                    if not laflechewallone.empty:
                        rank = laflechewallone['rank'].values[0].isdigit()
                        if rank:
                            result_laflechewallone = int(laflechewallone['rank'].values[0])

                    amstelgoldrace = df.loc[df['race_id'] == "amstel-gold-race_" + str(year)]
                    result_amstelgoldrace = np.nan
                    if not amstelgoldrace.empty:
                        rank = amstelgoldrace['rank'].values[0].isdigit()
                        if rank:
                            result_amstelgoldrace = int(amstelgoldrace['rank'].values[0])

                    sansebastian = df.loc[df['race_id'] == "san-sebastian_" + str(year)]
                    result_sansebastian = np.nan
                    if not sansebastian.empty:
                        rank = sansebastian['rank'].values[0].isdigit()
                        if rank:
                            result_sansebastian = int(sansebastian['rank'].values[0])

                    bretagneclassic = df.loc[df['race_id'] == "bretagne-classic_" + str(year)]
                    result_bretagneclassic = np.nan
                    if not bretagneclassic.empty:
                        rank = bretagneclassic['rank'].values[0].isdigit()
                        if rank:
                            result_bretagneclassic = int(bretagneclassic['rank'].values[0])

                    cyclassicshamburg = df.loc[df['race_id'] == "cyclassics-hamburg_" + str(year)]
                    result_cyclassicshamburg = np.nan
                    if not cyclassicshamburg.empty:
                        rank = cyclassicshamburg['rank'].values[0].isdigit()
                        if rank:
                            result_cyclassicshamburg = int(cyclassicshamburg['rank'].values[0])

                    gpquebec = df.loc[df['race_id'] == "gp-quebec_" + str(year)]
                    result_gpquebec = np.nan
                    if not gpquebec.empty:
                        rank = gpquebec['rank'].values[0].isdigit()
                        if rank:
                            result_gpquebec = int(gpquebec['rank'].values[0])

                    gpmontreal = df.loc[df['race_id'] == "gp-montreal_" + str(year)]
                    result_gpmontreal = np.nan
                    if not gpmontreal.empty:
                        rank = gpmontreal['rank'].values[0].isdigit()
                        if rank:
                            result_gpmontreal = int(gpmontreal['rank'].values[0])

                    # compare rider slug in rider metadata to get the correct row
                    # from row take birthday and nationality, then do calculations/abbreviations
                    # meta_riders = rider_meta_df['rider_slug'].tolist()
                    meta_rider = rider_meta_df.loc[rider_meta_df['rider_slug'] == rider_slug]
                    nationality = meta_rider['nationality'].values[0]
                    birthday = meta_rider['birthday'].values[0]
                    birthday = birthday.split(' ')[0]
                    birthday_dateobject = datetime.strptime(birthday, "%Y-%m-%d")
                    current_year = str(year) + '-12-31'
                    current_year_dateobject = datetime.strptime(current_year, "%Y-%m-%d")
                    rider_age = calc_age(birthday_dateobject, current_year_dateobject)
                    country = coco.convert(names=nationality, to='ISO2').lower()

                    rider_weight = float(meta_rider['weight'].values[0])
                    rider_height = float(meta_rider['height'].values[0])
                    rider_bmi = rider_weight / pow(rider_height, 2)

                    # rel_finish_time = df["time_rel"] #too many missing data, see later
                    in_top_20 = 0
                    actual_rank = 999
                    previous_pcs_points, previous_rank, previous2_rank, previous2_pcs_points = 0, 0, 0, 0

                    # rider is top 100 in current year but not always top 100 for all previous other years then scrape his data for that year
                    if (rider_slug not in prev_riders) or (rider_slug not in prev2_riders):
                        past_years = [years[-5], years[-4], years[-3], years[-2]]  # years
                        for season_year in past_years:
                            season = str(season_year) + '-12-31'
                            season_dateobject = datetime.strptime(season, "%Y-%m-%d")
                            age = calc_age(birthday_dateobject, season_dateobject)
                            target_top_riders_info = downloader.get_specific_top_riders(season_year, age, country)

                            results = target_top_riders_info.get('results')
                            if results.empty:  # if data empty = rider wasnt active that year skip that year
                                continue
                            riders = results['rider_slug'].tolist()
                            if rider_slug not in riders:  # if rider not in list, rider wasnt active that year skip that year
                                continue
                            rider = results.loc[results['rider_slug'] == rider_slug]

                            if season_year == years[-2]:
                                previous_pcs_points = float(rider['pcs_points'].values[0])
                                previous_rank = int(rider['rank'].values[0])

                            if season_year == years[-3]:
                                previous2_pcs_points = float(rider['pcs_points'].values[0])
                                previous2_rank = int(rider['rank'].values[0])

                        next_season = str(years[-1] + 1) + '-12-31'
                        next_season_dateobject = datetime.strptime(next_season, "%Y-%m-%d")
                        label_age = calc_age(birthday_dateobject, next_season_dateobject)
                        target_top_riders_info = downloader.get_specific_top_riders(years[-1] + 1, label_age, country)
                        results = target_top_riders_info.get('results')
                        if results.empty:  # if data empty = rider wasnt active that year skip that year
                            in_top_20 = 0
                            actual_rank = 999
                        elif rider_slug not in results[
                            'rider_slug'].tolist():  # if rider not in list, rider wasnt active that year skip that year
                            in_top_20 = 0
                            actual_rank = 999
                        else:
                            rider = results.loc[results['rider_slug'] == rider_slug]
                            actual_rank = int(rider['rank'].values[0])
                            if actual_rank <= 20:
                                in_top_20 = 1

                    else:
                        label = label_df.loc[label_df['rider_slug'] == rider_slug]
                        in_top_20 = 0
                        if not label.empty:
                            actual_rank = label['rank'].values[0]
                            if actual_rank <= 20:
                                in_top_20 = 1

                        previous2 = prev2_df.loc[prev2_df['rider_slug'] == rider_slug]
                        previous2_rank = int(previous2['rank'].values[0])
                        previous2_pcs_points = float(previous2['pcs_points'].values[0])

                        previous = prev_df.loc[prev_df['rider_slug'] == rider_slug]
                        previous_pcs_points = float(previous['pcs_points'].values[0])
                        previous_rank = int(previous['rank'].values[0])

                    current = current_df.loc[current_df['rider_slug'] == rider_slug]
                    current_pcs_points = float(current['pcs_points'].values[0])
                    current_rank = int(current['rank'].values[0])

                    # if rank or points is not 0 otherwise divide by 2 or 1 depening on number of 0's
                    rankpointslist = [current_rank, previous_rank, previous2_rank, current_pcs_points,
                                      previous_pcs_points, previous2_pcs_points]
                    numofzeroes = rankpointslist.count(0)

                    rank_avg_last3years = (current_rank + previous_rank + previous2_rank) / (3 - numofzeroes)
                    pcs_points_avg_last3years = (current_pcs_points + previous_pcs_points + previous2_pcs_points) / (
                                3 - numofzeroes)

                    rank_stdev_last3years = statistics.stdev([previous2_rank, previous_rank, current_rank])
                    pcs_points_stdev_last3years = statistics.stdev(
                        [previous2_pcs_points, previous_pcs_points, current_pcs_points])
                    win_ratio_stdev_last3years = statistics.stdev([win_ratio_na_prev2, win_ratio_na_prev, win_ratio_na])
                    top3_ratio_stdev_last3years = statistics.stdev(
                        [top3_ratio_na_prev2, top3_ratio_na_prev, top3_ratio_na])
                    top5_ratio_stdev_last3years = statistics.stdev(
                        [top5_ratio_na_prev2, top5_ratio_na_prev, top5_ratio_na])
                    top10_ratio_stdev_last3years = statistics.stdev(
                        [top10_ratio_na_prev2, top10_ratio_na_prev, top10_ratio_na])
                    stage_win_ratio_stdev_last3years = statistics.stdev(
                        [stage_win_ratio_na_prev2, stage_win_ratio_na_prev, stage_win_ratio_na])
                    stage_top3_ratio_stdev_last3years = statistics.stdev(
                        [stage_top3_ratio_na_prev2, stage_top3_ratio_na_prev, stage_top3_ratio_na])
                    stage_top5_ratio_stdev_last3years = statistics.stdev(
                        [stage_top5_ratio_na_prev2, stage_top5_ratio_na_prev, stage_top5_ratio_na])
                    stage_top10_ratio_stdev_last3years = statistics.stdev(
                        [stage_top10_ratio_na_prev2, stage_top10_ratio_na_prev, stage_top10_ratio_na])

                    df_row = {"rider_name": rider_slug, "year": year,
                              "Number_of_participated_one_day_races_current_year": number_races_na,
                              "Number_of_participated_multi_day_races_current_year": number_stage_races_na,
                              "Number_of_participated_one_day_races_all_time": number_races,
                              "Number_of_participated_multi_day_races_all_time": number_stage_races,
                              "Number_of_participated_one_day_races_last3years": number_races_last3years,
                              "Number_of_participated_multi_day_races_last3years": number_stage_races_last3years,
                              "win_ratio_all_time": win_ratio,
                              "top3_ratio_all_time": top3_ratio, "top5_ratio_all_time": top5_ratio,
                              "top10_ratio_all_time": top10_ratio,
                              "stage_win_ratio_all_time": stage_win_ratio,
                              "stage_top3_ratio_all_time": stage_top3_ratio,
                              "stage_top5_ratio_all_time": stage_top5_ratio,
                              "stage_top10_ratio_all_time": stage_top10_ratio,
                              "win_ratio": win_ratio_na,
                              "top3_ratio": top3_ratio_na, "top5_ratio": top5_ratio_na, "top10_ratio": top10_ratio_na,
                              "stage_win_ratio": stage_win_ratio_na, "stage_top3_ratio": stage_top3_ratio_na,
                              "stage_top5_ratio": stage_top5_ratio_na, "stage_top10_ratio": stage_top10_ratio_na,
                              "win_ratio_last3years": win_ratio_last3years,
                              "top3_ratio_last3years": top3_ratio_last3years,
                              "top5_ratio_last3years": top5_ratio_last3years,
                              "top10_ratio_last3years": top10_ratio_last3years,
                              "stage_win_ratio_last3years": stage_win_ratio_last3years,
                              "stage_top3_ratio_last3years": stage_top3_ratio_last3years,
                              "stage_top5_ratio_last3years": stage_top5_ratio_last3years,
                              "stage_top10_ratio_last3years": stage_top10_ratio_last3years,
                              "pcs_rank_previous_year": previous_rank, "pcs_rank_current_year": current_rank,
                              "pcs_points_previous_year": previous_pcs_points,
                              "pcs_points_current_year": current_pcs_points,
                              "rank_avg_last3years": rank_avg_last3years,
                              "pcs_points_avg_last3years": pcs_points_avg_last3years,
                              "win_ratio_slope": win_ratio_slope, "stage_win_ratio_slope": stage_win_ratio_slope,
                              "top3_ratio_slope": top3_ratio_slope, "top5_ratio_slope": top5_ratio_slope,
                              "top10_ratio_slope": top10_ratio_slope, "stage_top3_ratio_slope": stage_top3_ratio_slope,
                              "stage_top5_ratio_slope": stage_top5_ratio_slope,
                              "stage_top10_ratio_slope": stage_top10_ratio_slope,
                              "rank_stdev_last3years": rank_stdev_last3years,
                              "pcs_points_stdev_last3years": pcs_points_stdev_last3years,
                              "win_ratio_stdev_last3years": win_ratio_stdev_last3years,
                              "top3_ratio_stdev_last3years": top3_ratio_stdev_last3years,
                              "top5_ratio_stdev_last3years": top5_ratio_stdev_last3years,
                              "top10_ratio_stdev_last3years": top10_ratio_stdev_last3years,
                              "stage_win_ratio_stdev_last3years": stage_win_ratio_stdev_last3years,
                              "stage_top3_ratio_stdev_last3years": stage_top3_ratio_stdev_last3years,
                              "stage_top5_ratio_stdev_last3years": stage_top5_ratio_stdev_last3years,
                              "stage_top10_ratio_stdev_last3years": stage_top10_ratio_stdev_last3years,
                              "age": rider_age, "career year": career_year, "rider_bmi": rider_bmi,
                              "result_tour-de-france": result_tourdefrance,
                              "result_giro-d-italia": result_giroditalia, "result_vuelta-a-espana": result_vueltaespana,
                              "result_tour-down-under": result_tourdownunder, "result_paris-nice": result_parisnice,
                              "result_tirreno-adriatico": result_tirrenoadriatico,
                              "result_volta-a-catalunya": result_voltaacatalunya,
                              "result_itzulia-basque-country": result_itzuliabasquecountry,
                              "result_tour-de-romandie": result_tourderomandie,
                              "result_dauphine": result_dauphine, "result_tour-de-suisse": result_tourdesuisse,
                              "result_tour-de-pologne": result_tourdepologne,
                              "result_benelux_tour": result_beneluxtour, "result_milano-sanremo": result_milanosanremo,
                              "result_ronde-van-vlaanderen": result_rondevanvlaanderen,
                              "result_paris-roubaix": result_parisroubaix,
                              "result_liege-bastogne-liege": result_liegebastogneliege,
                              "result_il-lombardia": result_illombardia,
                              "result_strade-bianche": result_stradebianche,
                              "result_great-ocean-race": result_greatoceanrace,
                              "result_e3-harelbeke": result_e3harelbeke, "result_gent-wevelgem": result_gentwevelgem,
                              "result_la-fleche-wallone": result_laflechewallone,
                              "result_amstel-gold-race": result_amstelgoldrace,
                              "result_san-sebastian": result_sansebastian,
                              "result_bretagne-classic": result_bretagneclassic,
                              "result_cyclassics-hamburg": result_cyclassicshamburg,
                              "result_gp-quebec": result_gpquebec,
                              "result_gp-montreal": result_gpmontreal, "actual_rank": actual_rank,
                              "in_top20_next_year": in_top_20}
                    df_all = df_all.append(df_row, ignore_index=True)
        if years[-1] == 2021:  # here
            pass
        else:
            years.append(years[-1] + 1)

# to csv
all_data = df_all.to_csv("df_all.csv", index=False)
