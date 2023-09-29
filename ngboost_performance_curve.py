from ngboost import NGBRegressor, NGBClassifier, NGBoost
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import metrics
import time
import pickle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
import numpy as np
import matplotlib.pyplot as plt

def KNNAlterNGB(df_all, df_imputed, num_neighbours):
    # flat one day
    scaler = MinMaxScaler()

    flat_one_day = df_imputed[
        ["Number_of_participated_one_day_races_current_year", "Number_of_participated_multi_day_races_current_year",
         "win_ratio_all_time", "top3_ratio_all_time", "top5_ratio_all_time", "top10_ratio_all_time",
         "stage_win_ratio_all_time", "stage_top3_ratio_all_time", "stage_top5_ratio_all_time",
         "stage_top10_ratio_all_time",
         "win_ratio", "top3_ratio", "top5_ratio", "top10_ratio", "stage_win_ratio", "stage_top3_ratio",
         "stage_top5_ratio", "stage_top10_ratio", "pcs_rank_previous_year", "pcs_rank_current_year",
         "pcs_points_previous_year", "pcs_points_current_year",
         "rank_avg_last3years", "pcs_points_avg_last3years",
         "age", "career year","rider_bmi",
         "win_ratio_last3years", "top3_ratio_last3years", "top5_ratio_last3years", "top10_ratio_last3years",
         "stage_win_ratio_last3years", "stage_top3_ratio_last3years", "stage_top5_ratio_last3years",
         "stage_top10_ratio_last3years",
         "Number_of_participated_one_day_races_all_time", "Number_of_participated_multi_day_races_all_time",
         "Number_of_participated_one_day_races_last3years", "Number_of_participated_multi_day_races_last3years",
         "rank_stdev_last3years", "pcs_points_stdev_last3years",
         "win_ratio_stdev_last3years", "top3_ratio_stdev_last3years",
         "top5_ratio_stdev_last3years", "top10_ratio_stdev_last3years",
         "stage_win_ratio_stdev_last3years", "stage_top3_ratio_stdev_last3years",
         "stage_top5_ratio_stdev_last3years", "stage_top10_ratio_stdev_last3years",
         "result_milano-sanremo", "result_cyclassics-hamburg", "RankY1","RankY2","RankY3","RankY4","RankY5"]]

    flat_one_day = pd.DataFrame(scaler.fit_transform(flat_one_day), columns=flat_one_day.columns)  # normalize
    imputer = KNNImputer(n_neighbors=num_neighbours)
    flat_one_day = pd.DataFrame(imputer.fit_transform(flat_one_day), columns=flat_one_day.columns)  # impute
    flat_one_day = pd.DataFrame(scaler.inverse_transform(flat_one_day), columns=flat_one_day.columns)  # renormalize

    # hilly one day
    hilly_one_day = df_imputed[
        ["Number_of_participated_one_day_races_current_year", "Number_of_participated_multi_day_races_current_year",
         "win_ratio_all_time", "top3_ratio_all_time", "top5_ratio_all_time", "top10_ratio_all_time",
         "stage_win_ratio_all_time", "stage_top3_ratio_all_time", "stage_top5_ratio_all_time",
         "stage_top10_ratio_all_time",
         "win_ratio", "top3_ratio", "top5_ratio", "top10_ratio", "stage_win_ratio", "stage_top3_ratio",
         "stage_top5_ratio", "stage_top10_ratio", "pcs_rank_previous_year", "pcs_rank_current_year",
         "pcs_points_previous_year", "pcs_points_current_year",
         "rank_avg_last3years", "pcs_points_avg_last3years",
         "age", "career year","rider_bmi",
         "win_ratio_last3years", "top3_ratio_last3years", "top5_ratio_last3years", "top10_ratio_last3years",
         "stage_win_ratio_last3years", "stage_top3_ratio_last3years", "stage_top5_ratio_last3years",
         "stage_top10_ratio_last3years",
         "Number_of_participated_one_day_races_all_time", "Number_of_participated_multi_day_races_all_time",
         "Number_of_participated_one_day_races_last3years", "Number_of_participated_multi_day_races_last3years",
         "rank_stdev_last3years", "pcs_points_stdev_last3years",
         "win_ratio_stdev_last3years", "top3_ratio_stdev_last3years",
         "top5_ratio_stdev_last3years", "top10_ratio_stdev_last3years",
         "stage_win_ratio_stdev_last3years", "stage_top3_ratio_stdev_last3years",
         "stage_top5_ratio_stdev_last3years", "stage_top10_ratio_stdev_last3years",
         "result_liege-bastogne-liege", "result_il-lombardia", "result_strade-bianche",
         "result_great-ocean-race",
         "result_la-fleche-wallone", "result_amstel-gold-race", "result_san-sebastian",
         "result_bretagne-classic",
         "result_gp-quebec", "result_gp-montreal", "RankY1","RankY2","RankY3","RankY4","RankY5"]]

    hilly_one_day = pd.DataFrame(scaler.fit_transform(hilly_one_day), columns=hilly_one_day.columns)  # normalize
    imputer = KNNImputer(n_neighbors=num_neighbours)
    hilly_one_day = pd.DataFrame(imputer.fit_transform(hilly_one_day), columns=hilly_one_day.columns)  # impute
    hilly_one_day = pd.DataFrame(scaler.inverse_transform(hilly_one_day),
                                 columns=hilly_one_day.columns)  # renormalize

    # cobble classics
    cobble_classics = df_imputed[
        ["Number_of_participated_one_day_races_current_year", "Number_of_participated_multi_day_races_current_year",
         "win_ratio_all_time", "top3_ratio_all_time", "top5_ratio_all_time", "top10_ratio_all_time",
         "stage_win_ratio_all_time", "stage_top3_ratio_all_time", "stage_top5_ratio_all_time",
         "stage_top10_ratio_all_time",
         "win_ratio", "top3_ratio", "top5_ratio", "top10_ratio", "stage_win_ratio", "stage_top3_ratio",
         "stage_top5_ratio", "stage_top10_ratio", "pcs_rank_previous_year", "pcs_rank_current_year",
         "pcs_points_previous_year", "pcs_points_current_year",
         "rank_avg_last3years", "pcs_points_avg_last3years",
         "age", "career year","rider_bmi",
         "win_ratio_last3years", "top3_ratio_last3years", "top5_ratio_last3years", "top10_ratio_last3years",
         "stage_win_ratio_last3years", "stage_top3_ratio_last3years", "stage_top5_ratio_last3years",
         "stage_top10_ratio_last3years",
         "Number_of_participated_one_day_races_all_time", "Number_of_participated_multi_day_races_all_time",
         "Number_of_participated_one_day_races_last3years", "Number_of_participated_multi_day_races_last3years",
         "rank_stdev_last3years", "pcs_points_stdev_last3years",
         "win_ratio_stdev_last3years", "top3_ratio_stdev_last3years",
         "top5_ratio_stdev_last3years", "top10_ratio_stdev_last3years",
         "stage_win_ratio_stdev_last3years", "stage_top3_ratio_stdev_last3years",
         "stage_top5_ratio_stdev_last3years", "stage_top10_ratio_stdev_last3years",
         "result_ronde-van-vlaanderen", "result_paris-roubaix", "result_e3-harelbeke",
         "result_gent-wevelgem", "RankY1","RankY2","RankY3","RankY4","RankY5"]]
    cobble_classics = pd.DataFrame(scaler.fit_transform(cobble_classics),
                                   columns=cobble_classics.columns)  # normalize
    imputer = KNNImputer(n_neighbors=num_neighbours)
    cobble_classics = pd.DataFrame(imputer.fit_transform(cobble_classics),
                                   columns=cobble_classics.columns)  # impute
    cobble_classics = pd.DataFrame(scaler.inverse_transform(cobble_classics),
                                   columns=cobble_classics.columns)  # renormalize

    # multi day_races
    multi_day_races = df_imputed[
        ["Number_of_participated_one_day_races_current_year", "Number_of_participated_multi_day_races_current_year",
         "win_ratio_all_time", "top3_ratio_all_time", "top5_ratio_all_time", "top10_ratio_all_time",
         "stage_win_ratio_all_time", "stage_top3_ratio_all_time", "stage_top5_ratio_all_time",
         "stage_top10_ratio_all_time",
         "win_ratio", "top3_ratio", "top5_ratio", "top10_ratio", "stage_win_ratio", "stage_top3_ratio",
         "stage_top5_ratio", "stage_top10_ratio", "pcs_rank_previous_year", "pcs_rank_current_year",
         "pcs_points_previous_year", "pcs_points_current_year",
         "rank_avg_last3years", "pcs_points_avg_last3years",
         "age", "career year","rider_bmi",
         "win_ratio_last3years", "top3_ratio_last3years", "top5_ratio_last3years", "top10_ratio_last3years",
         "stage_win_ratio_last3years", "stage_top3_ratio_last3years", "stage_top5_ratio_last3years",
         "stage_top10_ratio_last3years",
         "Number_of_participated_one_day_races_all_time", "Number_of_participated_multi_day_races_all_time",
         "Number_of_participated_one_day_races_last3years", "Number_of_participated_multi_day_races_last3years",
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
         "result_tour-de-pologne",
        "RankY1","RankY2","RankY3","RankY4","RankY5"]]
    multi_day_races = pd.DataFrame(scaler.fit_transform(multi_day_races),
                                   columns=multi_day_races.columns)  # normalize
    imputer = KNNImputer(n_neighbors=num_neighbours)
    multi_day_races = pd.DataFrame(imputer.fit_transform(multi_day_races),
                                   columns=multi_day_races.columns)  # impute
    multi_day_races = pd.DataFrame(scaler.inverse_transform(multi_day_races),
                                   columns=multi_day_races.columns)  # renormalize

    # slope
    slope = df_imputed[
        ["Number_of_participated_one_day_races_current_year", "Number_of_participated_multi_day_races_current_year",
         "win_ratio_all_time", "top3_ratio_all_time", "top5_ratio_all_time", "top10_ratio_all_time",
         "stage_win_ratio_all_time", "stage_top3_ratio_all_time", "stage_top5_ratio_all_time",
         "stage_top10_ratio_all_time",
         "win_ratio", "top3_ratio", "top5_ratio", "top10_ratio", "stage_win_ratio", "stage_top3_ratio",
         "stage_top5_ratio", "stage_top10_ratio", "pcs_rank_previous_year", "pcs_rank_current_year",
         "pcs_points_previous_year", "pcs_points_current_year",
         "rank_avg_last3years", "pcs_points_avg_last3years",
         "age", "career year","rider_bmi",
         "win_ratio_last3years", "top3_ratio_last3years", "top5_ratio_last3years", "top10_ratio_last3years",
         "stage_win_ratio_last3years", "stage_top3_ratio_last3years", "stage_top5_ratio_last3years",
         "stage_top10_ratio_last3years",
         "Number_of_participated_one_day_races_all_time", "Number_of_participated_multi_day_races_all_time",
         "Number_of_participated_one_day_races_last3years", "Number_of_participated_multi_day_races_last3years",
         "rank_stdev_last3years", "pcs_points_stdev_last3years",
         "win_ratio_stdev_last3years", "top3_ratio_stdev_last3years",
         "top5_ratio_stdev_last3years", "top10_ratio_stdev_last3years",
         "stage_win_ratio_stdev_last3years", "stage_top3_ratio_stdev_last3years",
         "stage_top5_ratio_stdev_last3years", "stage_top10_ratio_stdev_last3years",
         "win_ratio_slope", "stage_win_ratio_slope", "top3_ratio_slope",
         "top5_ratio_slope", "top10_ratio_slope",
         "stage_top3_ratio_slope", "stage_top5_ratio_slope",
         "stage_top10_ratio_slope",
         "RankY1","RankY2","RankY3","RankY4","RankY5"]]
    slope = pd.DataFrame(scaler.fit_transform(slope), columns=slope.columns)  # normalize
    imputer = KNNImputer(n_neighbors=num_neighbours)
    slope = pd.DataFrame(imputer.fit_transform(slope), columns=slope.columns)  # impute
    slope = pd.DataFrame(scaler.inverse_transform(slope), columns=slope.columns)  # renormalize

    # drop year and add original year, add rider name
    # df_imputed = df_imputed.drop(columns=['year'])
    riders = df_all['rider_name']
    years = df_all['year']
    flat_one_day = flat_one_day[["result_milano-sanremo", "result_cyclassics-hamburg"]]
    hilly_one_day = hilly_one_day[
        ["result_liege-bastogne-liege", "result_il-lombardia", "result_strade-bianche", "result_great-ocean-race",
         "result_la-fleche-wallone", "result_amstel-gold-race", "result_san-sebastian", "result_bretagne-classic",
         "result_gp-quebec", "result_gp-montreal"]]
    cobble_classics = cobble_classics[
        ["result_ronde-van-vlaanderen", "result_paris-roubaix", "result_e3-harelbeke", "result_gent-wevelgem"]]
    multi_day_races = multi_day_races[
        ["result_tour-de-france", "result_giro-d-italia", "result_vuelta-a-espana", "result_tour-down-under",
         "result_paris-nice", "result_tirreno-adriatico", "result_volta-a-catalunya",
         "result_itzulia-basque-country",
         "result_tour-de-romandie", "result_dauphine", "result_tour-de-suisse", "result_tour-de-pologne"
         ]]

    frames = [riders, years, slope, flat_one_day, hilly_one_day, cobble_classics, multi_day_races]
    df_imputed = pd.concat(frames, axis=1)

    df_imputed = df_imputed.astype({"Number_of_participated_one_day_races_current_year": 'int',
                                    "Number_of_participated_multi_day_races_current_year": 'int',
                                    "pcs_rank_previous_year": 'int',"pcs_rank_current_year": 'int',
                                    "age": 'int', "career year": 'int',
                                    "Number_of_participated_one_day_races_all_time": 'int',
                                    "Number_of_participated_multi_day_races_all_time": 'int',
                                    "Number_of_participated_one_day_races_last3years": 'int',
                                    "Number_of_participated_multi_day_races_last3years": 'int',
                                    "result_tour-de-france": 'int', "result_giro-d-italia": 'int', "result_vuelta-a-espana": 'int',
                                    "result_tour-down-under": 'int',
                                    "result_paris-nice": 'int', "result_tirreno-adriatico": 'int', "result_volta-a-catalunya": 'int',
                                    "result_itzulia-basque-country": 'int',
                                    "result_tour-de-romandie": 'int', "result_dauphine": 'int', "result_tour-de-suisse": 'int',
                                    "result_tour-de-pologne": 'int',
                                    "result_milano-sanremo": 'int', "result_ronde-van-vlaanderen": 'int', "result_paris-roubaix": 'int',
                                    "result_liege-bastogne-liege": 'int',
                                    "result_il-lombardia": 'int', "result_strade-bianche": 'int', "result_great-ocean-race": 'int',
                                    "result_e3-harelbeke": 'int', "result_gent-wevelgem": 'int',
                                    "result_la-fleche-wallone": 'int', "result_amstel-gold-race": 'int', "result_san-sebastian": 'int',
                                    "result_bretagne-classic": 'int',
                                    "result_cyclassics-hamburg": 'int', "result_gp-quebec": 'int', "result_gp-montreal": 'int',
                                    "RankY1": 'int', "RankY2": 'int', "RankY3": 'int', "RankY4": 'int', "RankY5": 'int'
                                   })
    return df_imputed

def feature_selection(X_train,X_valid,X_test,y_train):
    # feature_selection

    feature_names = list(X_train.columns.values)
    #selector = SelectKBest(score_func=mutual_info_classif, k=30)
    #X_train = selector.fit_transform(X_train, y_train)
    #mask = selector.get_support()  # list of booleans
    #new_features = []
    #removed_features = []
    #for bool, feature in zip(mask, feature_names):
    #    if bool:
    #        new_features.append(feature)
    #    else:
    #        removed_features.append(feature)
    #X_train = pd.DataFrame(X_train, columns=new_features)
    ## remove all feature in xvalid and xtest that are not in xtrain anymore
    #X_valid = X_valid.drop(columns=removed_features)
    #X_test = X_test.drop(columns=removed_features)

    #return X_train,X_valid,X_test,new_features

    return X_train, X_valid, X_test, feature_names


class NGBModel():
    def __init__(self):
        # read_csv
        self.df_curve = pd.read_csv("df_all_curve.csv",
                             usecols=["rider_name", "year", "Number_of_participated_one_day_races_current_year",
                                      "Number_of_participated_multi_day_races_current_year",
                                      "win_ratio_all_time", "top3_ratio_all_time", "top5_ratio_all_time",
                                      "top10_ratio_all_time",
                                      "stage_win_ratio_all_time", "stage_top3_ratio_all_time",
                                      "stage_top5_ratio_all_time", "stage_top10_ratio_all_time",
                                      "win_ratio", "top3_ratio", "top5_ratio", "top10_ratio", "stage_win_ratio",
                                      "stage_top3_ratio",
                                      "stage_top5_ratio", "stage_top10_ratio", "pcs_rank_previous_year",
                                      "pcs_rank_current_year",
                                      "pcs_points_previous_year", "pcs_points_current_year",
                                      "rank_avg_last3years", "pcs_points_avg_last3years",
                                      "age", "career year","win_ratio_slope","rider_bmi",
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
                                      "result_tour-de-pologne",
                                      "result_milano-sanremo", "result_ronde-van-vlaanderen", "result_paris-roubaix",
                                      "result_liege-bastogne-liege",
                                      "result_il-lombardia", "result_strade-bianche", "result_great-ocean-race",
                                      "result_e3-harelbeke", "result_gent-wevelgem",
                                      "result_la-fleche-wallone", "result_amstel-gold-race", "result_san-sebastian",
                                      "result_bretagne-classic",
                                      "result_cyclassics-hamburg", "result_gp-quebec", "result_gp-montreal",
                                      "RankY1","RankY2","RankY3","RankY4","RankY5"])


        self.df_final_result = pd.DataFrame([],
                                       columns=["rider_name", "year", "Number_of_participated_one_day_races_current_year",
                                                "Number_of_participated_multi_day_races_current_year",
                                                "win_ratio_all_time", "top3_ratio_all_time", "top5_ratio_all_time",
                                                "top10_ratio_all_time",
                                                "stage_win_ratio_all_time", "stage_top3_ratio_all_time",
                                                "stage_top5_ratio_all_time", "stage_top10_ratio_all_time",
                                                "win_ratio", "top3_ratio", "top5_ratio", "top10_ratio", "stage_win_ratio",
                                                "stage_top3_ratio",
                                                "stage_top5_ratio", "stage_top10_ratio", "pcs_rank_previous_year",
                                                "pcs_rank_current_year",
                                                "pcs_points_previous_year", "pcs_points_current_year",
                                                "rank_avg_last3years", "pcs_points_avg_last3years",
                                                "age", "career year", "win_ratio_slope","rider_bmi",
                                                "stage_win_ratio_slope", "top3_ratio_slope", "top5_ratio_slope",
                                                "top10_ratio_slope",
                                                "stage_top3_ratio_slope", "stage_top5_ratio_slope",
                                                "stage_top10_ratio_slope",
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
                                                "result_tour-de-pologne",
                                                "result_milano-sanremo", "result_ronde-van-vlaanderen",
                                                "result_paris-roubaix",
                                                "result_liege-bastogne-liege",
                                                "result_il-lombardia", "result_strade-bianche", "result_great-ocean-race",
                                                "result_e3-harelbeke", "result_gent-wevelgem",
                                                "result_la-fleche-wallone", "result_amstel-gold-race",
                                                "result_san-sebastian",
                                                "result_bretagne-classic",
                                                "result_cyclassics-hamburg", "result_gp-quebec", "result_gp-montreal",
                                                "RankY1","RankY2","RankY3","RankY4","RankY5"])

        # metric csv
        self.df_metric_results = pd.DataFrame([], columns=["model", "RMSE", "MAE", "MAPE"])

        self.df_metric_yearly_results = pd.DataFrame([], columns=["model", "year", "RMSE", "MAE", "MAPE"])

        self.best_ngb_paramY1 = {
            'boosting_rounds': 0,
            'learning_rate': 0,
            'col_sample':0,
            'minibatch_frac':0,
            'RMSE': 0,
            'RMSE_train': 0,
            'MAE':0,
            'MAE_train':0,
            'MAPE':0,
            'MAPE_train':0,
            'kneighbour': 0,
            'PredInRange': 0,
            'PredInRange_train': 0
        }
        self.best_ngb_paramY2 = {
            'boosting_rounds': 0,
            'learning_rate': 0,
            'col_sample': 0,
            'minibatch_frac': 0,
            'RMSE': 0,
            'RMSE_train': 0,
            'MAE':0,
            'MAE_train':0,
            'MAPE':0,
            'MAPE_train':0,
            'kneighbour': 0,
            'PredInRange': 0,
            'PredInRange_train': 0
        }
        self.best_ngb_paramY3 = {
            'boosting_rounds': 0,
            'learning_rate': 0,
            'col_sample': 0,
            'minibatch_frac': 0,
            'RMSE': 0,
            'RMSE_train': 0,
            'MAE':0,
            'MAE_train':0,
            'MAPE':0,
            'MAPE_train':0,
            'kneighbour': 0,
            'PredInRange': 0,
            'PredInRange_train': 0
        }
        self.best_ngb_paramY4 = {
            'boosting_rounds': 0,
            'learning_rate': 0,
            'col_sample': 0,
            'minibatch_frac': 0,
            'RMSE': 0,
            'RMSE_train': 0,
            'MAE':0,
            'MAE_train':0,
            'MAPE':0,
            'MAPE_train':0,
            'kneighbour': 0,
            'PredInRange': 0,
            'PredInRange_train': 0
        }
        self.best_ngb_paramY5 = {
            'boosting_rounds': 0,
            'learning_rate': 0,
            'col_sample': 0,
            'minibatch_frac': 0,
            'RMSE': 0,
            'RMSE_train': 0,
            'MAE':0,
            'MAE_train':0,
            'MAPE':0,
            'MAPE_train':0,
            'kneighbour': 0,
            'PredInRange': 0,
            'PredInRange_train': 0
        }


    def calculatePredInRange(self,y,y_pred,y_pred_std):
        lower_bounds = []
        upper_bounds = []
        for predrank, stdev in zip(y_pred, y_pred_std):
            lower_bound = predrank - stdev
            lower_bounds.append(lower_bound)
            upper_bound = predrank + stdev
            upper_bounds.append(upper_bound)

        within_range = 0
        for lower, upper, actual in zip(lower_bounds, upper_bounds, y):
            if lower <= actual <= upper:
                within_range += 1

        percentage_within_range = (within_range / len(y)) * 100
        return percentage_within_range
        
        
    def trainNGB(self,X_train,X_valid,y_train,y_valid,ngb_param,rounds,lr,k,col_sample,mini_batch):

        ngb_reg = NGBRegressor(n_estimators=rounds, learning_rate=lr, natural_gradient=True,
                               random_state=42,col_sample=col_sample,minibatch_frac=mini_batch,verbose=0)

        ngb_reg.fit(X_train, y_train)
        y_pred_train = ngb_reg.predict(X_train)
        y_pred_valid = ngb_reg.predict(X_valid)

        y_pred_dist_train = ngb_reg.pred_dist(X_train) #training set distribution
        y_pred_std_train = y_pred_dist_train.std()

        y_pred_dist_valid = ngb_reg.pred_dist(X_valid)
        y_pred_std_valid = y_pred_dist_valid.std()

        # y_pred_mean = y_pred_dist_valid.mean()
        # y_pred_mean_train = y_pred_dist_train.mean()

        if (metrics.mean_absolute_error(y_valid, y_pred_valid) < ngb_param['MAE']) or ngb_param['MAE'] == 0:
            RMSE_ngb_valid_year = metrics.mean_squared_error(y_valid, y_pred_valid,squared=False)
            RMSE_ngb_train_year = metrics.mean_squared_error(y_train,y_pred_train,squared=False)
            MAE_ngb_valid_year = metrics.mean_absolute_error(y_valid,y_pred_valid)
            MAE_ngb_train_year = metrics.mean_absolute_error(y_train,y_pred_train)
            MAPE_ngb_valid_year = metrics.mean_absolute_percentage_error(y_valid,y_pred_valid)
            MAPE_ngb_train_year = metrics.mean_absolute_percentage_error(y_train, y_pred_train)
            
            pred_in_range_train_year = self.calculatePredInRange(y_train,y_pred_train,y_pred_std_train)
            pred_in_range_valid_year = self.calculatePredInRange(y_valid,y_pred_valid,y_pred_std_valid)

            best_ngb_param = {
                'learning_rate': lr,
                'boosting_rounds': rounds,
                'col_sample': col_sample,
                'minibatch_frac': mini_batch,
                'RMSE': RMSE_ngb_valid_year,
                'RMSE_train': RMSE_ngb_train_year,
                'MAE': MAE_ngb_valid_year,
                'MAE_train': MAE_ngb_train_year,
                'MAPE': MAPE_ngb_valid_year,
                'MAPE_train': MAPE_ngb_train_year,
                'kneighbour': k,
                'PredInRange': pred_in_range_valid_year,
                'PredInRange_train': pred_in_range_train_year
            }
            return best_ngb_param
        return ngb_param


    def split_into_sets(self,df_imputed,train_years):
        df_train = df_imputed.loc[df_imputed['year'].isin(train_years)]
        df_valid = df_imputed.loc[df_imputed['year'] == train_years[-1] + 1]
        df_test = df_imputed.loc[df_imputed['year'] == train_years[-1] + 2]
        return df_train,df_valid,df_test

    def predictTestSet(self,X_train,X_valid,X_test,y_train,y_valid,rounds,lr,col_sample,mini_batch,year):
        frames = [X_train, X_valid]

        xtv = pd.concat(frames)
        frames = [y_train, y_valid]
        ytv = pd.concat(frames)
        ngb_reg = NGBRegressor(n_estimators=rounds,
                               learning_rate=lr, natural_gradient=True,random_state=42,col_sample=col_sample,minibatch_frac=mini_batch,verbose=0)
        ngb_reg.fit(xtv, ytv)

        pickle.dump(ngb_reg, open('saved_models/ngb'+year+'_model.sav', 'wb'))

        y_pred = ngb_reg.predict(X_test)
        y_pred_dist = ngb_reg.pred_dist(X_test)  # test set distribution
        y_pred_mean = y_pred_dist.mean()
        y_pred_std = y_pred_dist.std()
        
        return y_pred,y_pred_mean,y_pred_std


    def plot_performance_curve(self,year,num_top_riders):
        df_results = pd.read_csv("df_results/df_results_NGB.csv",
                                    usecols=["rider_name", "year", "pcs_rank_current_year",
                                             "RankY1", "RankY2", "RankY3", "RankY4", "RankY5",
                                             "Y1Prediction", "Y1mean", "Y1std", "Y2Prediction", "Y2mean", "Y2std",
                                             "Y3Prediction", "Y3mean", "Y3std", "Y4Prediction", "Y4mean", "Y4std",
                                             "Y5Prediction", "Y5mean", "Y5std"
                                             ])

        #filter by year and top riders
        df = df_results.loc[(df_results['year'] == year) & (df_results['pcs_rank_current_year'] <= num_top_riders)]

        years = []
        for i in range(year+1,year+6):
            years.append(i)
        x = np.array(years)

        for index, row in df.iterrows():
            pred_ranks = [row['Y1Prediction'],row['Y2Prediction'],row['Y3Prediction'],row['Y4Prediction'],row['Y5Prediction']]
            y = np.array(pred_ranks)

            std_devs = [row['Y1std'], row['Y2std'], row['Y3std'], row['Y4std'],
                          row['Y5std']]
            std_dev = np.array(std_devs)

            actual_ranks = [row['RankY1'],row['RankY2'],row['RankY3'],row['RankY4'],row['RankY5']]
            y2 = np.array(actual_ranks)


            #plot predicted rank
            plt.plot(x, y, color='blue', marker='o', linestyle='-', label='Predicted Rank')

            # Plot error bars representing the standard deviation
            plt.errorbar(x, y, yerr=std_dev, fmt='none', ecolor='red', elinewidth=1, capsize=4, label='Standard Deviation')

            #plot actual rank
            plt.plot(x, y2, color='green', marker='o', linestyle='-', label='Actual Rank')

            # Add labels and legend
            plt.title(row['rider_name']+" performance curve " + str(year))
            plt.xlabel('year')
            plt.ylabel('rank')
            plt.legend()

            #save plot
            plt.savefig('performance_curve/'+row['rider_name']+str(year)+"_"+str(row["pcs_rank_current_year"]))
            plt.clf()
            # Display the plot
            #plt.show()


    def plot_performance_curve_specific_rider(self,rider_name):
        # another function with specific rider name gives all curves for all years for that rider
        df_results = pd.read_csv("df_results/df_results_NGB.csv",
                                    usecols=["rider_name", "year", "pcs_rank_current_year",
                                             "RankY1", "RankY2", "RankY3", "RankY4", "RankY5",
                                             "Y1Prediction", "Y1mean", "Y1std", "Y2Prediction", "Y2mean", "Y2std",
                                             "Y3Prediction", "Y3mean", "Y3std", "Y4Prediction", "Y4mean", "Y4std",
                                             "Y5Prediction", "Y5mean", "Y5std"
                                             ])

        #filter by year and top riders
        df = df_results.loc[df_results['rider_name'] == rider_name]

        for index, row in df.iterrows():
            years = []
            for i in range(row['year'] + 1, row['year'] + 6):
                years.append(i)
            x = np.array(years)

            pred_ranks = [row['Y1Prediction'],row['Y2Prediction'],row['Y3Prediction'],row['Y4Prediction'],row['Y5Prediction']]
            y = np.array(pred_ranks)

            std_devs = [row['Y1std'], row['Y2std'], row['Y3std'], row['Y4std'],
                          row['Y5std']]
            std_dev = np.array(std_devs)

            actual_ranks = [row['RankY1'],row['RankY2'],row['RankY3'],row['RankY4'],row['RankY5']]
            y2 = np.array(actual_ranks)


            #plot predicted rank
            plt.plot(x, y, color='blue', marker='o', linestyle='-', label='Predicted Rank')

            # Plot error bars representing the standard deviation
            plt.errorbar(x, y, yerr=std_dev, fmt='none', ecolor='red', elinewidth=1, capsize=4, label='Standard Deviation')

            #plot actual rank
            plt.plot(x, y2, color='green', marker='o', linestyle='-', label='Actual Rank')

            # Add labels and legend
            plt.title(row['rider_name']+" performance curve "+ str(row["year"]))
            plt.xlabel('year')
            plt.ylabel('rank')
            plt.legend()

            #save plot
            plt.savefig('performance_curve/'+row['rider_name']+str(row['year'])+"_"+str(row["pcs_rank_current_year"]))
            plt.clf()
            # Display the plot
            #plt.show()

    def run(self):
        start = time.time()


        #turn fselect off maybe and all parameters/neighbours back on
        boosting_round_ngb = [100,400] #[100,200,300,400,500]
        learning_rate_ngb = [0.1] #[0.1, 0.3, 0.5, 0.7, 1.0]
        col_sample_ngb = [0.3, 0.7, 1.0] #[0.1, 0.3, 0.5, 0.7, 1.0]
        minibatch_frac_ngb = [1.0] #[0.5, 0.75, 1.0]

        MAE_ngb_train= 0
        MAE_ngb_valid= 0
        MAE_ngb_test= 0
        MAPE_ngb_train = 0
        MAPE_ngb_valid = 0
        MAPE_ngb_test = 0
        RMSE_ngb_train = 0
        RMSE_ngb_valid = 0
        RMSE_ngb_test = 0
        PredInRange_ngb_train = 0
        PredInRange_ngb_valid = 0
        PredInRange_ngb_test = 0

        filename = 'saved_models/ngb_model.sav'


        train_years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]

        count = 1

        numofneighbours = [1,3,7,9] #numofneighbours = [1,3,5,7,9,11]

        #self.df_curve['5_year_ranks'] = self.df_curve['5_year_ranks'].apply(ast.literal_eval)  # convert the string column back to a list

        df_imputed = self.df_curve.drop(columns=['rider_name', 'year'])
        df_bestk = pd.DataFrame([], columns=["yearmodel", "bestkY1","bestkY2","bestkY3","bestkY4","bestkY5"])

        for year in train_years:
            df_result = pd.DataFrame([],
                                     columns=["rider_name", "year", "Number_of_participated_one_day_races_current_year",
                                              "Number_of_participated_multi_day_races_current_year",
                                              "win_ratio_all_time", "top3_ratio_all_time", "top5_ratio_all_time",
                                              "top10_ratio_all_time",
                                              "stage_win_ratio_all_time", "stage_top3_ratio_all_time",
                                              "stage_top5_ratio_all_time", "stage_top10_ratio_all_time",
                                              "win_ratio", "top3_ratio", "top5_ratio", "top10_ratio", "stage_win_ratio",
                                              "stage_top3_ratio",
                                              "stage_top5_ratio", "stage_top10_ratio", "pcs_rank_previous_year",
                                              "pcs_rank_current_year",
                                              "pcs_points_previous_year", "pcs_points_current_year",
                                              "rank_avg_last3years", "pcs_points_avg_last3years",
                                              "age", "career year", "win_ratio_slope","rider_bmi",
                                              "stage_win_ratio_slope", "top3_ratio_slope", "top5_ratio_slope",
                                              "top10_ratio_slope",
                                              "stage_top3_ratio_slope", "stage_top5_ratio_slope",
                                              "stage_top10_ratio_slope",
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
                                              "result_paris-nice", "result_tirreno-adriatico",
                                              "result_volta-a-catalunya", "result_itzulia-basque-country",
                                              "result_tour-de-romandie", "result_dauphine", "result_tour-de-suisse",
                                              "result_tour-de-pologne",
                                              "result_milano-sanremo", "result_ronde-van-vlaanderen",
                                              "result_paris-roubaix", "result_liege-bastogne-liege",
                                              "result_il-lombardia", "result_strade-bianche", "result_great-ocean-race",
                                              "result_e3-harelbeke", "result_gent-wevelgem",
                                              "result_la-fleche-wallone", "result_amstel-gold-race",
                                              "result_san-sebastian", "result_bretagne-classic",
                                              "result_cyclassics-hamburg", "result_gp-quebec", "result_gp-montreal",
                                              "RankY1","RankY2","RankY3","RankY4","RankY5"])

            for k in numofneighbours:
                df_imputed = KNNAlterNGB(self.df_curve, df_imputed, k)
                # all_data = df_imputed.to_csv("df_imputed.csv", index=False)

                df_train = df_imputed.loc[df_imputed['year'].isin(train_years)]
                df_valid = df_imputed.loc[df_imputed['year'] == train_years[-1] + 1]
                df_test = df_imputed.loc[df_imputed['year'] == train_years[-1] + 2]
                
                X_train1, y_train1 = df_train.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_train[
                                       "RankY1"].copy()
                X_valid1, y_valid1 = df_valid.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_valid[
                                       "RankY1"].copy()
                X_test1, y_test1 = df_test.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_test[
                                       "RankY1"].copy()
                
                X_train2, y_train2 = df_train.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_train[
                                       "RankY2"].copy()
                X_valid2, y_valid2 = df_valid.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_valid[
                                       "RankY2"].copy()
                X_test2, y_test2 = df_test.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_test[
                                       "RankY2"].copy()
                
                X_train3, y_train3 = df_train.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_train[
                                       "RankY3"].copy()
                X_valid3, y_valid3 = df_valid.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_valid[
                                       "RankY3"].copy()
                X_test3, y_test3 = df_test.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_test[
                                       "RankY3"].copy()
                
                X_train4, y_train4 = df_train.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_train[
                                       "RankY4"].copy()
                X_valid4, y_valid4 = df_valid.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_valid[
                                       "RankY4"].copy()
                X_test4, y_test4 = df_test.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_test[
                                       "RankY4"].copy()
                
                X_train5, y_train5 = df_train.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_train[
                                       "RankY5"].copy()
                X_valid5, y_valid5 = df_valid.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_valid[
                                       "RankY5"].copy()
                X_test5, y_test5 = df_test.drop(
                    columns=["RankY1","RankY2","RankY3","RankY4","RankY5", 'year', 'rider_name']).copy(), df_test[
                                       "RankY5"].copy()
    
                X_train1,X_valid1,X_test1,new_features1 = feature_selection(X_train1,X_valid1,X_test1,y_train1)
                X_train2,X_valid2,X_test2,new_features2 = feature_selection(X_train2,X_valid2,X_test2,y_train2)
                X_train3,X_valid3,X_test3,new_features3 = feature_selection(X_train3,X_valid3,X_test3,y_train3)
                X_train4,X_valid4,X_test4,new_features4 = feature_selection(X_train4,X_valid4,X_test4,y_train4)
                X_train5,X_valid5,X_test5,new_features5 = feature_selection(X_train5,X_valid5,X_test5,y_train5)

                for lr in learning_rate_ngb:
                    for rounds in boosting_round_ngb:
                        for col_sample in col_sample_ngb:
                            for minibatch_frac in minibatch_frac_ngb:
                                self.best_ngb_paramY1 = self.trainNGB(X_train1,X_valid1,y_train1,y_valid1,self.best_ngb_paramY1,rounds,lr,k,col_sample,minibatch_frac)
                                self.best_ngb_paramY2 = self.trainNGB(X_train2,X_valid2,y_train2,y_valid2,self.best_ngb_paramY2,rounds,lr,k,col_sample,minibatch_frac)
                                self.best_ngb_paramY3 = self.trainNGB(X_train3,X_valid3,y_train3,y_valid3,self.best_ngb_paramY3,rounds,lr,k,col_sample,minibatch_frac)
                                self.best_ngb_paramY4 = self.trainNGB(X_train4,X_valid4,y_train4,y_valid4,self.best_ngb_paramY4,rounds,lr,k,col_sample,minibatch_frac)
                                self.best_ngb_paramY5 = self.trainNGB(X_train5,X_valid5,y_train5,y_valid5,self.best_ngb_paramY5,rounds,lr,k,col_sample,minibatch_frac)
                                #self.best_ngb_paramY1 = self.trainNGB(X_train1, X_valid1, y_train1, y_valid1,
                                #                                      self.best_ngb_paramY1, 200, lr, 7, col_sample,
                                #                                      minibatch_frac)
                                #self.best_ngb_paramY2 = self.trainNGB(X_train2, X_valid2, y_train2, y_valid2,
                                #                                      self.best_ngb_paramY2, 300, lr, 1, col_sample,
                                #                                      minibatch_frac)
                                #self.best_ngb_paramY3 = self.trainNGB(X_train3, X_valid3, y_train3, y_valid3,
                                #                                      self.best_ngb_paramY3, 200, lr, 1, col_sample,
                                #                                      minibatch_frac)
                                #self.best_ngb_paramY4 = self.trainNGB(X_train4, X_valid4, y_train4, y_valid4,
                                #                                      self.best_ngb_paramY4, 300, lr, 9, col_sample,
                                #                                      minibatch_frac)
                                #self.best_ngb_paramY5 = self.trainNGB(X_train5, X_valid5, y_train5, y_valid5,
                                #                                      self.best_ngb_paramY5, 100, lr, 11, col_sample,
                                #                                      minibatch_frac)
            df_imputed = self.df_curve.drop(columns=['rider_name', 'year'])
            df_imputed1 = KNNAlterNGB(self.df_curve, df_imputed, self.best_ngb_paramY1["kneighbour"])
            df_imputed2 = KNNAlterNGB(self.df_curve, df_imputed, self.best_ngb_paramY2["kneighbour"])
            df_imputed3 = KNNAlterNGB(self.df_curve, df_imputed, self.best_ngb_paramY3["kneighbour"])
            df_imputed4 = KNNAlterNGB(self.df_curve, df_imputed, self.best_ngb_paramY4["kneighbour"])
            df_imputed5 = KNNAlterNGB(self.df_curve, df_imputed, self.best_ngb_paramY5["kneighbour"])

            df_imputed1.to_csv("df_imputed/df_imputed_ngbY1.csv", index=False)
            df_imputed2.to_csv("df_imputed/df_imputed_ngbY2.csv", index=False)
            df_imputed3.to_csv("df_imputed/df_imputed_ngbY3.csv", index=False)
            df_imputed4.to_csv("df_imputed/df_imputed_ngbY4.csv", index=False)
            df_imputed5.to_csv("df_imputed/df_imputed_ngbY5.csv", index=False)

            df_train1,df_valid1,df_test1 = self.split_into_sets(df_imputed1,train_years)
            df_train2,df_valid2,df_test2 = self.split_into_sets(df_imputed2,train_years)
            df_train3,df_valid3,df_test3 = self.split_into_sets(df_imputed3,train_years)
            df_train4,df_valid4,df_test4 = self.split_into_sets(df_imputed4,train_years)
            df_train5,df_valid5,df_test5 = self.split_into_sets(df_imputed5,train_years)

            X_train1, y_train1 = df_train1.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_train1[
                                     "RankY1"].copy()
            X_valid1, y_valid1 = df_valid1.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_valid1[
                                     "RankY1"].copy()
            X_test1, y_test1 = df_test1.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_test1[
                                   "RankY1"].copy()

            X_train2, y_train2 = df_train2.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_train2[
                                     "RankY2"].copy()
            X_valid2, y_valid2 = df_valid2.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_valid2[
                                     "RankY2"].copy()
            X_test2, y_test2 = df_test2.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_test2[
                                   "RankY2"].copy()

            X_train3, y_train3 = df_train3.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_train3[
                                     "RankY3"].copy()
            X_valid3, y_valid3 = df_valid3.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_valid3[
                                     "RankY3"].copy()
            X_test3, y_test3 = df_test3.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_test3[
                                   "RankY3"].copy()

            X_train4, y_train4 = df_train4.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_train4[
                                     "RankY4"].copy()
            X_valid4, y_valid4 = df_valid4.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_valid4[
                                     "RankY4"].copy()
            X_test4, y_test4 = df_test4.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_test4[
                                   "RankY4"].copy()

            X_train5, y_train5 = df_train5.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_train5[
                                     "RankY5"].copy()
            X_valid5, y_valid5 = df_valid5.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_valid5[
                                     "RankY5"].copy()
            X_test5, y_test5 = df_test5.drop(
                columns=["RankY1", "RankY2", "RankY3", "RankY4", "RankY5", 'year', 'rider_name']).copy(), df_test5[
                                   "RankY5"].copy()

            # feature_selection
            X_train1, X_valid1, X_test1, new_features1 = feature_selection(X_train1, X_valid1, X_test1, y_train1)
            X_train2, X_valid2, X_test2, new_features2 = feature_selection(X_train2, X_valid2, X_test2, y_train2)
            X_train3, X_valid3, X_test3, new_features3 = feature_selection(X_train3, X_valid3, X_test3, y_train3)
            X_train4, X_valid4, X_test4, new_features4 = feature_selection(X_train4, X_valid4, X_test4, y_train4)
            X_train5, X_valid5, X_test5, new_features5 = feature_selection(X_train5, X_valid5, X_test5, y_train5)


            y_predY1,y_pred_meanY1,y_pred_stdY1 = self.predictTestSet(X_train1, X_valid1, X_test1, y_train1, y_valid1, self.best_ngb_paramY1["boosting_rounds"],self.best_ngb_paramY1["learning_rate"],self.best_ngb_paramY1["col_sample"],self.best_ngb_paramY1["minibatch_frac"],"Y1")
            y_predY2,y_pred_meanY2,y_pred_stdY2 = self.predictTestSet(X_train2, X_valid2, X_test2, y_train2, y_valid2, self.best_ngb_paramY2["boosting_rounds"],self.best_ngb_paramY2["learning_rate"],self.best_ngb_paramY2["col_sample"],self.best_ngb_paramY2["minibatch_frac"],"Y2")
            y_predY3,y_pred_meanY3,y_pred_stdY3 = self.predictTestSet(X_train3, X_valid3, X_test3, y_train3, y_valid3, self.best_ngb_paramY3["boosting_rounds"],self.best_ngb_paramY3["learning_rate"],self.best_ngb_paramY3["col_sample"],self.best_ngb_paramY3["minibatch_frac"],"Y3")
            y_predY4,y_pred_meanY4,y_pred_stdY4 = self.predictTestSet(X_train4, X_valid4, X_test4, y_train4, y_valid4, self.best_ngb_paramY4["boosting_rounds"],self.best_ngb_paramY4["learning_rate"],self.best_ngb_paramY4["col_sample"],self.best_ngb_paramY4["minibatch_frac"],"Y4")
            y_predY5,y_pred_meanY5,y_pred_stdY5 = self.predictTestSet(X_train5, X_valid5, X_test5, y_train5, y_valid5, self.best_ngb_paramY5["boosting_rounds"],self.best_ngb_paramY5["learning_rate"],self.best_ngb_paramY5["col_sample"],self.best_ngb_paramY5["minibatch_frac"],"Y5")

            # prepare df beforehand and then compare dfall and xtest,ypred and append to df and then write to csv
            # row and y_pred to csv with their names included so gotta compare with df_imputed
            # iterate through dataframe:
            for index, row in X_test1.iterrows():
                df_row = df_imputed1.loc[
                    (df_imputed1[new_features1[0]] == row[new_features1[0]])
                    & (df_imputed1[new_features1[1]] == row[
                        new_features1[1]])
                    & (df_imputed1[new_features1[2]] == row[new_features1[2]]) & (
                            df_imputed1[new_features1[3]] == row[new_features1[3]])
                    & (df_imputed1[new_features1[4]] == row[new_features1[4]]) & (
                            df_imputed1[new_features1[5]] == row[new_features1[5]])
                    ]

                df_result = df_result.append(df_row, ignore_index=True)
            # add columns mean,std,pred_rank,y1-y5
            pred1 = pd.Series(y_predY1, name='Y1Prediction')
            mean1 = pd.Series(y_pred_meanY1, name='Y1mean')
            std1 = pd.Series(y_pred_stdY1, name='Y1std')

            pred2 = pd.Series(y_predY2, name='Y2Prediction')
            mean2 = pd.Series(y_pred_meanY2, name='Y2mean')
            std2 = pd.Series(y_pred_stdY2, name='Y2std')

            pred3 = pd.Series(y_predY3, name='Y3Prediction')
            mean3 = pd.Series(y_pred_meanY3, name='Y3mean')
            std3 = pd.Series(y_pred_stdY3, name='Y3std')

            pred4 = pd.Series(y_predY4, name='Y4Prediction')
            mean4 = pd.Series(y_pred_meanY4, name='Y4mean')
            std4 = pd.Series(y_pred_stdY4, name='Y4std')

            pred5 = pd.Series(y_predY5, name='Y5Prediction')
            mean5 = pd.Series(y_pred_meanY5, name='Y5mean')
            std5 = pd.Series(y_pred_stdY5, name='Y5std')

            frames = [df_result,pred1,mean1,std1,pred2,mean2,std2,pred3,mean3,std3,pred4,mean4,std4,pred5,mean5,std5]
            df_result = pd.concat(frames, axis=1)
            self.df_final_result = self.df_final_result.append(df_result)

            MAE_ngb_train_year = (self.best_ngb_paramY1['MAE_train'] + self.best_ngb_paramY2['MAE_train']
                                       + self.best_ngb_paramY3['MAE_train'] + self.best_ngb_paramY4['MAE_train'] +
                                       self.best_ngb_paramY5['MAE_train']) / 5

            MAE_ngb_valid_year = (self.best_ngb_paramY1['MAE'] + self.best_ngb_paramY2['MAE']
            + self.best_ngb_paramY3['MAE'] + self.best_ngb_paramY4['MAE'] + self.best_ngb_paramY5['MAE'])/5

            MAE_ngb_test_year = (metrics.mean_absolute_error(y_test1, y_predY1) + metrics.mean_absolute_error(y_test2, y_predY2) +
                 metrics.mean_absolute_error(y_test3, y_predY3) + metrics.mean_absolute_error(y_test4, y_predY4) +
                 metrics.mean_absolute_error(y_test5, y_predY5))/5
            
            MAE_ngb_train += MAE_ngb_train_year
            MAE_ngb_valid += MAE_ngb_valid_year
            MAE_ngb_test += MAE_ngb_test_year

            RMSE_ngb_train_year = (self.best_ngb_paramY1['RMSE_train'] + self.best_ngb_paramY2['RMSE_train']
                                  + self.best_ngb_paramY3['RMSE_train'] + self.best_ngb_paramY4['RMSE_train'] +
                                  self.best_ngb_paramY5['RMSE_train']) / 5

            RMSE_ngb_valid_year = (self.best_ngb_paramY1['RMSE'] + self.best_ngb_paramY2['RMSE']
                                  + self.best_ngb_paramY3['RMSE'] + self.best_ngb_paramY4['RMSE'] + self.best_ngb_paramY5[
                                      'RMSE']) / 5

            RMSE_ngb_test_year = (metrics.mean_squared_error(y_test1, y_predY1,squared=False) + metrics.mean_squared_error(y_test2,
                                                                                                              y_predY2,squared=False) +
                                 metrics.mean_squared_error(y_test3, y_predY3,squared=False) + metrics.mean_squared_error(y_test4,
                                                                                                              y_predY4,squared=False) +
                                 metrics.mean_squared_error(y_test5, y_predY5,squared=False)) / 5

            RMSE_ngb_train += RMSE_ngb_train_year
            RMSE_ngb_valid += RMSE_ngb_valid_year
            RMSE_ngb_test += RMSE_ngb_test_year

            MAPE_ngb_train_year = (self.best_ngb_paramY1['MAPE_train'] + self.best_ngb_paramY2['MAPE_train']
                                  + self.best_ngb_paramY3['MAPE_train'] + self.best_ngb_paramY4['MAPE_train'] +
                                  self.best_ngb_paramY5['MAPE_train']) / 5

            MAPE_ngb_valid_year = (self.best_ngb_paramY1['MAPE'] + self.best_ngb_paramY2['MAPE']
                                  + self.best_ngb_paramY3['MAPE'] + self.best_ngb_paramY4['MAPE'] + self.best_ngb_paramY5['MAPE']) / 5

            MAPE_ngb_test_year = (metrics.mean_absolute_percentage_error(y_test1, y_predY1) + metrics.mean_absolute_percentage_error(y_test2,y_predY2) +
                                 metrics.mean_absolute_percentage_error(y_test3, y_predY3) + metrics.mean_absolute_percentage_error(y_test4,y_predY4) +
                                 metrics.mean_absolute_percentage_error(y_test5, y_predY5)) / 5

            MAPE_ngb_train += MAPE_ngb_train_year
            MAPE_ngb_valid += MAPE_ngb_valid_year
            MAPE_ngb_test += MAPE_ngb_test_year
            
            
            PredInRange_ngb_train_year = (self.best_ngb_paramY1['PredInRange_train'] + self.best_ngb_paramY2['PredInRange_train']
                                  + self.best_ngb_paramY3['PredInRange_train'] + self.best_ngb_paramY4['PredInRange_train'] +
                                  self.best_ngb_paramY5['PredInRange_train']) / 5

            PredInRange_ngb_valid_year = (self.best_ngb_paramY1['PredInRange'] + self.best_ngb_paramY2['PredInRange']
                                  + self.best_ngb_paramY3['PredInRange'] + self.best_ngb_paramY4['PredInRange'] + self.best_ngb_paramY5[
                                      'PredInRange']) / 5

            PredInRange_ngb_test_year = (self.calculatePredInRange(y_test1,y_predY1,y_pred_stdY1) + self.calculatePredInRange(y_test2,y_predY2,y_pred_stdY2)
                                         + self.calculatePredInRange(y_test3,y_predY3,y_pred_stdY3) + self.calculatePredInRange(y_test4,y_predY4,y_pred_stdY4)
                                         + self.calculatePredInRange(y_test5,y_predY5,y_pred_stdY5)) / 5

            PredInRange_ngb_train += PredInRange_ngb_train_year
            PredInRange_ngb_valid += PredInRange_ngb_valid_year
            PredInRange_ngb_test += PredInRange_ngb_test_year

            
            df_row = {"yearmodel": train_years[-1] + 3, "bestkY1": self.best_ngb_paramY1["kneighbour"],
                      "bestkY2": self.best_ngb_paramY2["kneighbour"],"bestkY3": self.best_ngb_paramY3["kneighbour"],
                      "bestkY4": self.best_ngb_paramY4["kneighbour"],"bestkY5": self.best_ngb_paramY5["kneighbour"]}
            df_bestk = df_bestk.append(df_row, ignore_index=True)

            df_row = {"model": "NGBoost Training set", "year": train_years[-1] + 3,
                      "MAE": MAE_ngb_train_year,"RMSE": RMSE_ngb_train_year,"MAPE": MAPE_ngb_train_year,"PredInRange": PredInRange_ngb_train_year}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            df_row = {"model": "NGBoost Validation set", "year": train_years[-1] + 3,
                      "MAE": MAE_ngb_valid_year,"RMSE": RMSE_ngb_valid_year,"MAPE": MAPE_ngb_valid_year,"PredInRange": PredInRange_ngb_valid_year}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            df_row = {"model": "NGBoost Test set", "year": train_years[-1] + 3,
                      "MAE": MAE_ngb_test_year,"RMSE": RMSE_ngb_test_year,"MAPE": MAPE_ngb_test_year,"PredInRange": PredInRange_ngb_test_year}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)

            self.df_metric_yearly_results.to_csv("metric_yearly_results/df_metric_yearly_results_ngb.csv", index=False)

            if train_years[-1] == 2015: #here
                break
            else:
                train_years.append(train_years[-1] + 1)
                count += 1

        all_data = df_bestk.to_csv("df_bestk/df_bestk_NGB.csv", index=False)

        all_data = self.df_final_result.to_csv("df_results/df_results_NGB.csv", index=False)

        MAE_ngb_train = MAE_ngb_train / count
        MAE_ngb_valid = MAE_ngb_valid / count
        MAE_ngb_test = MAE_ngb_test / count

        RMSE_ngb_train = RMSE_ngb_train / count
        RMSE_ngb_valid = RMSE_ngb_valid / count
        RMSE_ngb_test = RMSE_ngb_test / count

        MAPE_ngb_train = MAPE_ngb_train / count
        MAPE_ngb_valid = MAPE_ngb_valid / count
        MAPE_ngb_test = MAPE_ngb_test / count

        PredInRange_ngb_train = PredInRange_ngb_train / count
        PredInRange_ngb_valid = PredInRange_ngb_valid / count
        PredInRange_ngb_test = PredInRange_ngb_test / count

        end = time.time()
        print(end - start)
        print('done')

        df_row = {"model": "NGBoost Training set",
                  "MAE": MAE_ngb_train,"RMSE": RMSE_ngb_train,"MAPE": MAPE_ngb_train,"PredInRange": PredInRange_ngb_train}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        df_row = {"model": "NGBoost Validation set",
                  "MAE": MAE_ngb_valid,"RMSE": RMSE_ngb_valid,"MAPE": MAPE_ngb_valid,"PredInRange": PredInRange_ngb_valid}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        df_row = {"model": "NGBoost Test set",
                  "MAE": MAE_ngb_test,"RMSE": RMSE_ngb_test,"MAPE": MAPE_ngb_test,"PredInRange": PredInRange_ngb_test}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)

        self.df_metric_results.to_csv("metric_results/df_metric_results_ngb.csv", index=False)

        print(self.best_ngb_paramY1)
        print(self.best_ngb_paramY2)
        print(self.best_ngb_paramY3)
        print(self.best_ngb_paramY4)
        print(self.best_ngb_paramY5)

