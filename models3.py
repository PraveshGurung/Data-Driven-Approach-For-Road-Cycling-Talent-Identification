import numpy as np
import pandas as pd
import glob
import xgboost as xgb
import catboost as cgb
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time
import shap  # package used to calculate Shap values
import matplotlib.pyplot as plt
import pickle
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
####knn-imputation#####
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


def KNNAlter(df_all, df_imputed, num_neighbours):
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
         "result_milano-sanremo", "result_cyclassics-hamburg", "in_top20_next_year"]]
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
         "result_gp-quebec", "result_gp-montreal", "in_top20_next_year"]]
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
         "result_gent-wevelgem", "in_top20_next_year"]]
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
        "result_tour-de-pologne", "in_top20_next_year"]]
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
         "in_top20_next_year"]]
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
         "result_tour-de-romandie", "result_dauphine", "result_tour-de-suisse", "result_tour-de-pologne"]]

    frames = [riders, years, slope, flat_one_day, hilly_one_day, cobble_classics, multi_day_races]
    df_imputed = pd.concat(frames, axis=1)

    return df_imputed

class Model():
    def __init__(self):
        # read_csv
        self.df_all = pd.read_csv("df_all.csv",
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
                                      "result_tour-de-pologne",
                                      "result_milano-sanremo", "result_ronde-van-vlaanderen", "result_paris-roubaix",
                                      "result_liege-bastogne-liege",
                                      "result_il-lombardia", "result_strade-bianche", "result_great-ocean-race",
                                      "result_e3-harelbeke", "result_gent-wevelgem",
                                      "result_la-fleche-wallone", "result_amstel-gold-race", "result_san-sebastian",
                                      "result_bretagne-classic",
                                      "result_cyclassics-hamburg", "result_gp-quebec", "result_gp-montreal",
                                      "in_top20_next_year"])

        # metric csv
        self.df_metric_results = pd.DataFrame([],
                                         columns=["model", "logloss", "roc auc score", "f1_score", "MSE", "accuracy",
                                                  "precision", "recall", "pr auc score"])

        self.df_metric_yearly_results = pd.DataFrame([],
                                                columns=["model", "year", "logloss", "roc auc score", "f1_score", "MSE",
                                                         "accuracy",
                                                         "precision", "recall", "pr auc score"])

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
                                                "age", "career year","rider_bmi", "win_ratio_slope",
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
                                                "in_top20_next_year"])


class XGBoost(Model):
    def __init__(self):
        Model.__init__(self)
        self.best_xgb_param_logloss = {
            'alpha': 0,
            'boosting_rounds': 0,
            'max_depth': 0,
            'learning_rate': 0,
            'reg_lambda': 0,
            'subsample': 0,
            'colsample_bytree': 0,
            'gamma': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'objective': 'binary:logistic',
            'logloss': 0,
            'auc': 0,
            'MSE': 0,
            'accuracy': 0,
            'pr_auc_score': 0,
            'precision_train': 0,
            'recall_train': 0,
            'f1_score_train': 0,
            'logloss_train': 0,
            'auc_train': 0,
            'MSE_train': 0,
            'accuracy_train': 0,
            'pr_auc_score_train': 0,
            'kneighbour': 0
        }

    def run(self):
        start = time.time()

        alpha_xgb = [0]
        boosting_round_xgb = [300]
        max_depth_xgb = [10]
        learning_rate_xgb = [0.9]
        reg_lambda_xgb = [1]
        subsample_xgb = [1.0]
        colsample_bytree_xgb = [0.5]
        gamma_xgb = [12]

        auc_xgb_train, logloss_xgb_train, f1_scores_xgb_train, MSE_xgb_train, accuracy_xgb_train, precision_xgb_train, recall_xgb_train, pr_auc_xgb_train = 0, 0, 0, 0, 0, 0, 0, 0
        auc_xgb_valid, logloss_xgb_valid, f1_scores_xgb_valid, MSE_xgb_valid, accuracy_xgb_valid, precision_xgb_valid, recall_xgb_valid, pr_auc_xgb_valid = 0, 0, 0, 0, 0, 0, 0, 0
        auc_xgb_test, logloss_xgb_test, f1_scores_xgb_test, MSE_xgb_test, accuracy_xgb_test, precision_xgb_test, recall_xgb_test, pr_auc_xgb_test = 0, 0, 0, 0, 0, 0, 0, 0

        filename = 'saved_models/xgb_model.sav'

        train_years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
        count = 1

        numofneighbours = [1] #numofneighbours = [1, 3, 5, 7, 9, 11]

        df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
        df_bestk = pd.DataFrame([], columns=["yearmodel", "bestk"])

        for year in train_years:
            auc_xgb_valid_year, logloss_xgb_valid_year, f1_scores_xgb_valid_year, MSE_xgb_valid_year, accuracy_xgb_valid_year, \
            precision_xgb_valid_year, recall_xgb_valid_year, pr_auc_xgb_valid_year = 0, 0, 0, 0, 0, 0, 0, 0

            auc_xgb_train_year, logloss_xgb_train_year, f1_scores_xgb_train_year, MSE_xgb_train_year, accuracy_xgb_train_year, \
            precision_xgb_train_year, recall_xgb_train_year, pr_auc_xgb_train_year = 0, 0, 0, 0, 0, 0, 0, 0
            df_result = pd.DataFrame([], columns=["rider_name", "year", "Number_of_participated_one_day_races_current_year",
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
                                                  "result_tour-de-pologne",
                                                  "result_milano-sanremo", "result_ronde-van-vlaanderen",
                                                  "result_paris-roubaix", "result_liege-bastogne-liege",
                                                  "result_il-lombardia", "result_strade-bianche", "result_great-ocean-race",
                                                  "result_e3-harelbeke", "result_gent-wevelgem",
                                                  "result_la-fleche-wallone", "result_amstel-gold-race", "result_san-sebastian",
                                                  "result_bretagne-classic",
                                                  "result_cyclassics-hamburg", "result_gp-quebec", "result_gp-montreal",
                                                  "in_top20_next_year"])

            for k in numofneighbours:
                df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
                df_imputed = KNNAlter(self.df_all, df_imputed, k)
                # all_data = df_imputed.to_csv("df_imputed.csv", index=False)

                df_train = df_imputed.loc[df_imputed['year'].isin(train_years)]
                df_valid = df_imputed.loc[df_imputed['year'] == train_years[-1] + 1]
                df_test = df_imputed.loc[df_imputed['year'] == train_years[-1] + 2]
                X_train, y_train = df_train.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_train[
                                       'in_top20_next_year'].copy()
                X_valid, y_valid = df_valid.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_valid[
                                       'in_top20_next_year'].copy()
                X_test, y_test = df_test.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_test[
                                     'in_top20_next_year'].copy()

                # insert previous UPsampling SMOTE code here if needed
                oversample = SMOTE()
                X_train, y_train = oversample.fit_resample(X_train, y_train)

                # feature_selection
                feature_names = list(X_train.columns.values)
                selector = SelectKBest(score_func=mutual_info_classif, k=30)
                X_train = selector.fit_transform(X_train, y_train)  # k optimal?
                mask = selector.get_support()  # list of booleans
                new_features = []
                removed_features = []
                for bool, feature in zip(mask, feature_names):
                    if bool:
                        new_features.append(feature)
                    else:
                        removed_features.append(feature)
                X_train = pd.DataFrame(X_train, columns=new_features)

                # remove all feature in xvalid and xtest that are not in xtrain anymore
                X_valid = X_valid.drop(columns=removed_features)
                X_test = X_test.drop(columns=removed_features)

                for max_depth in max_depth_xgb:
                    for lr in learning_rate_xgb:
                        for rounds in boosting_round_xgb:
                            for alpha in alpha_xgb:
                                for gamma in gamma_xgb:
                                    for subsample in subsample_xgb:
                                        for landa in reg_lambda_xgb:
                                            for colsample_bytree in colsample_bytree_xgb:
                                                xg_clf = xgb.XGBClassifier(objective='binary:logistic',
                                                                           learning_rate=lr, max_depth=max_depth, alpha=alpha,
                                                                           n_estimators=rounds, reg_lambda=landa,
                                                                           use_label_encoder=False, gamma=gamma,
                                                                           subsample=subsample,
                                                                           colsample_bytree=colsample_bytree,
                                                                           eval_metric='error', tree_method='gpu_hist')

                                                xg_clf.fit(X_train, y_train)

                                                y_pred = xg_clf.predict(X_valid)
                                                y_pred_train = xg_clf.predict(X_train)

                                                if (metrics.f1_score(y_valid,
                                                                     y_pred) > f1_scores_xgb_valid_year) or f1_scores_xgb_valid_year == 0:
                                                    logloss_xgb_valid_year = metrics.log_loss(y_valid, y_pred)
                                                    fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred)
                                                    auc_xgb_valid_year = metrics.auc(fpr, tpr)
                                                    f1_scores_xgb_valid_year = metrics.f1_score(y_valid, y_pred)
                                                    MSE_xgb_valid_year = metrics.mean_squared_error(y_valid, y_pred)
                                                    accuracy_xgb_valid_year = metrics.accuracy_score(y_valid, y_pred)
                                                    precision_xgb_valid_year, recall_xgb_valid_year, fscore, support = metrics.precision_recall_fscore_support(
                                                        y_valid, y_pred, average='weighted')
                                                    xgb_probs = xg_clf.predict_proba(X_valid)
                                                    # keep probabilities for the positive outcome only
                                                    xgb_probs = xgb_probs[:, 1]
                                                    xgb_precision, xgb_recall, _ = metrics.precision_recall_curve(y_valid,
                                                                                                                  xgb_probs)
                                                    pr_auc_xgb_valid_year = metrics.auc(xgb_recall, xgb_precision)

                                                    logloss_xgb_train_year = metrics.log_loss(y_train, y_pred_train)
                                                    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_train)
                                                    auc_xgb_train_year = metrics.auc(fpr, tpr)
                                                    f1_scores_xgb_train_year = metrics.f1_score(y_train, y_pred_train)
                                                    MSE_xgb_train_year = metrics.mean_squared_error(y_train, y_pred_train)
                                                    accuracy_xgb_train_year = metrics.accuracy_score(y_train, y_pred_train)
                                                    precision_xgb_train_year, recall_xgb_train_year, fscore, support = metrics.precision_recall_fscore_support(
                                                        y_train, y_pred_train, average='weighted')
                                                    xgb_probs = xg_clf.predict_proba(X_train)
                                                    # keep probabilities for the positive outcome only
                                                    xgb_probs = xgb_probs[:, 1]
                                                    xgb_precision, xgb_recall, _ = metrics.precision_recall_curve(y_train,
                                                                                                                  xgb_probs)
                                                    pr_auc_xgb_train_year = metrics.auc(xgb_recall, xgb_precision)

                                                    self.best_xgb_param_logloss = {
                                                        'max_depth': max_depth,
                                                        'learning_rate': lr,
                                                        'boosting_rounds': rounds,
                                                        'alpha': alpha,
                                                        'gamma': gamma,
                                                        'colsample_bytree': colsample_bytree,
                                                        'subsample': subsample,
                                                        'reg_lambda': landa,
                                                        'logloss': logloss_xgb_valid_year,
                                                        'f1_score': f1_scores_xgb_valid_year,
                                                        'MSE': MSE_xgb_valid_year,
                                                        'auc': auc_xgb_valid_year,
                                                        'accuracy': accuracy_xgb_valid_year,
                                                        'precision': precision_xgb_valid_year,
                                                        'recall': recall_xgb_valid_year,
                                                        'pr_auc_score': pr_auc_xgb_valid_year,
                                                        'logloss_train': logloss_xgb_train_year,
                                                        'auc_train': auc_xgb_train_year,
                                                        'f1_score_train': f1_scores_xgb_train_year,
                                                        'MSE_train': MSE_xgb_train_year,
                                                        'accuracy_train': accuracy_xgb_train_year,
                                                        'precision_train': precision_xgb_train_year,
                                                        'recall_train': recall_xgb_train_year,
                                                        'pr_auc_score_train': pr_auc_xgb_train_year,
                                                        'kneighbour': k
                                                    }

            df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
            df_imputed = KNNAlter(self.df_all, df_imputed, self.best_xgb_param_logloss["kneighbour"])
            all_data = df_imputed.to_csv("df_imputed/df_imputed_XGB.csv", index=False)

            df_train = df_imputed.loc[df_imputed['year'].isin(train_years)]
            df_valid = df_imputed.loc[df_imputed['year'] == train_years[-1] + 1]
            df_test = df_imputed.loc[df_imputed['year'] == train_years[-1] + 2]
            X_train, y_train = df_train.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_train[
                                   'in_top20_next_year'].copy()
            X_valid, y_valid = df_valid.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_valid[
                                   'in_top20_next_year'].copy()
            X_test, y_test = df_test.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_test[
                                 'in_top20_next_year'].copy()

            # insert previous UPsampling SMOTE code here if needed
            oversample = SMOTE()
            X_train, y_train = oversample.fit_resample(X_train, y_train)

            # feature_selection
            feature_names = list(X_train.columns.values)
            selector = SelectKBest(score_func=mutual_info_classif, k=30)
            X_train = selector.fit_transform(X_train, y_train)  # k optimal?
            mask = selector.get_support()  # list of booleans
            new_features = []
            removed_features = []
            for bool, feature in zip(mask, feature_names):
                if bool:
                    new_features.append(feature)
                else:
                    removed_features.append(feature)
            X_train = pd.DataFrame(X_train, columns=new_features)

            # remove all feature in xvalid and xtest that are not in xtrain anymore
            X_valid = X_valid.drop(columns=removed_features)
            X_test = X_test.drop(columns=removed_features)

            frames = [X_train, X_valid]
            xtv = pd.concat(frames)
            frames = [y_train, y_valid]
            ytv = pd.concat(frames)
            xg_clf = xgb.XGBClassifier(objective='binary:logistic',
                                       learning_rate=self.best_xgb_param_logloss['learning_rate'],
                                       max_depth=self.best_xgb_param_logloss['max_depth'],
                                       alpha=self.best_xgb_param_logloss['alpha'],
                                       n_estimators=self.best_xgb_param_logloss['boosting_rounds'],
                                       subsample=self.best_xgb_param_logloss['subsample'], gamma=self.best_xgb_param_logloss['gamma'],
                                       colsample_bytree=self.best_xgb_param_logloss['colsample_bytree'],
                                       reg_lambda=self.best_xgb_param_logloss['reg_lambda'],
                                       use_label_encoder=False, eval_metric='error', tree_method='gpu_hist')
            xg_clf.fit(xtv, ytv)

            pickle.dump(xg_clf, open(filename, 'wb'))
            y_pred = xg_clf.predict(X_test)

            # check for different years test set
            if train_years[-1] == 2019: #here
                xg_clf = pickle.load(open(filename, 'rb'))

                explainer = shap.TreeExplainer(xg_clf)
                shap_values = explainer(X_test)
                # get the data row info so you can check which rider and what year it is
                # also manually choose which rider to analyse

                df_reset = df_test.reset_index()

                # riderlist = ["alejandro-valverde", "bauke-mollema", "david-gaudu", "egan-bernal", "jasper-philipsen",
                # "joao-almeida",
                # "jonas-vingegaard-rasmussen", "matej-mohoric", "michael-woods", "richard-carapaz",
                # "sonny-colbrelli", "tim-merlier"]
                riderlist = ["alejandro-valverde", "bauke-mollema"]

                # l = df_reset.index[df_reset.loc[df_reset['rider_name'].isin(riderlist)]].tolist()
                # l = df_reset.index[df_reset['rider_name'] == "fernando-gaviria"].tolist()
                # ['pcs_rank_current_year'] == 40

                l = []
                for i in riderlist:
                    l += df_reset.index[df_reset['rider_name'] == i].tolist()
                # ['pcs_rank_current_year'] == 40

                for i in l:
                    shap.plots.waterfall(shap_values[i], max_display=21)  # 1 sample

                # shap.plots.force(shap_values[5]) #1 sample basically same as waterfall
                expected_value = explainer.expected_value
                shap_array = explainer.shap_values(X_test)

                # Descion plot for first 10 observations
                # take index of all wrong prediction of top 20 and plot the decision plot?
                shap.decision_plot(expected_value, shap_array[0:10], feature_names=list(X_test.columns))  # x samples

                shap.plots.bar(shap_values)  # all samples

                path = 'save_path_here.png'
                shap.plots.beeswarm(shap_values, plot_size=1.8, max_display=21, show=False)  # all samples
                plt.savefig(path, bbox_inches='tight', dpi=300)

                # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
                # SHAP scatter plots
                # shap.plots.scatter(shap_values[:, "feature name"], ax=ax[0], show=False)
                # shap.plots.scatter(shap_values[:, "feature name"], ax=ax[1])

            # prepare df beforehand and then compare dfall and xtest,ypred and append to df and then write to csv
            # row and y_pred to csv with their names included so gotta compare with df_imputed
            # iterate through dataframe:
            for index, row in X_test.iterrows():
                df_row = df_imputed.loc[
                    (df_imputed[new_features[0]] == row[new_features[0]])
                    & (df_imputed[new_features[1]] == row[
                        new_features[1]])
                    & (df_imputed[new_features[2]] == row[new_features[2]]) & (
                            df_imputed[new_features[3]] == row[new_features[3]])
                    & (df_imputed[new_features[4]] == row[new_features[4]]) & (
                            df_imputed[new_features[5]] == row[new_features[5]])
                    ]

                df_result = df_result.append(df_row, ignore_index=True)
            pred = pd.Series(y_pred, name='Prediction')

            frames = [df_result, pred]
            df_result = pd.concat(frames, axis=1)
            self.df_final_result = self.df_final_result.append(df_result)

            f1_scores_xgb_valid += self.best_xgb_param_logloss['f1_score']
            precision_xgb_valid += self.best_xgb_param_logloss['precision']
            recall_xgb_valid += self.best_xgb_param_logloss['recall']
            auc_xgb_valid += self.best_xgb_param_logloss['auc']
            logloss_xgb_valid += self.best_xgb_param_logloss['logloss']
            MSE_xgb_valid += self.best_xgb_param_logloss['MSE']
            accuracy_xgb_valid += self.best_xgb_param_logloss['accuracy']
            pr_auc_xgb_valid += self.best_xgb_param_logloss['pr_auc_score']

            f1_scores_xgb_train += self.best_xgb_param_logloss['f1_score_train']
            precision_xgb_train += self.best_xgb_param_logloss['precision_train']
            recall_xgb_train += self.best_xgb_param_logloss['recall_train']
            auc_xgb_train += self.best_xgb_param_logloss['auc_train']
            logloss_xgb_train += self.best_xgb_param_logloss['logloss_train']
            MSE_xgb_train += self.best_xgb_param_logloss['MSE_train']
            accuracy_xgb_train += self.best_xgb_param_logloss['accuracy_train']
            pr_auc_xgb_train += self.best_xgb_param_logloss['pr_auc_score_train']

            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            auc_xgb_test_year = metrics.auc(fpr, tpr)
            logloss_xgb_test_year = metrics.log_loss(y_test, y_pred)
            MSE_xgb_test_year = metrics.mean_squared_error(y_test, y_pred)
            accuracy_xgb_test_year = metrics.accuracy_score(y_test, y_pred)
            f1_scores_xgb_test_year = metrics.f1_score(y_test, y_pred)
            precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, y_pred, average='weighted')

            auc_xgb_test += auc_xgb_test_year
            logloss_xgb_test += logloss_xgb_test_year
            MSE_xgb_test += MSE_xgb_test_year
            accuracy_xgb_test += accuracy_xgb_test_year
            f1_scores_xgb_test += f1_scores_xgb_test_year
            precision_xgb_test = precision + precision_xgb_test
            recall_xgb_test = recall + recall_xgb_test

            xgb_probs = xg_clf.predict_proba(X_test)
            # keep probabilities for the positive outcome only
            xgb_probs = xgb_probs[:, 1]
            # predict class values
            y_pred = xg_clf.predict(X_test)
            xgb_precision, xgb_recall, _ = metrics.precision_recall_curve(y_test, xgb_probs)
            pr_auc_xgb_test_year = metrics.auc(xgb_recall, xgb_precision)
            pr_auc_xgb_test += pr_auc_xgb_test_year

            df_row = {"yearmodel": train_years[-1] + 3, "bestk": self.best_xgb_param_logloss["kneighbour"]}
            df_bestk = df_bestk.append(df_row, ignore_index=True)

            df_row = {"model": "XGBoost Training set", "year": train_years[-1] + 3,
                      "f1_score": self.best_xgb_param_logloss['f1_score_train'],
                      "precision": self.best_xgb_param_logloss['precision_train'], "recall": self.best_xgb_param_logloss['recall_train'],
                      "logloss": self.best_xgb_param_logloss['logloss_train'], "roc auc score": self.best_xgb_param_logloss['auc_train'],
                      "MSE": self.best_xgb_param_logloss['MSE_train'], "accuracy": self.best_xgb_param_logloss['accuracy_train'],
                      "pr auc score": self.best_xgb_param_logloss['pr_auc_score_train']}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            df_row = {"model": "XGBoost Validation set", "year": train_years[-1] + 3,
                      "f1_score": self.best_xgb_param_logloss['f1_score'],
                      "precision": self.best_xgb_param_logloss['precision'], "recall": self.best_xgb_param_logloss['recall'],
                      "logloss": self.best_xgb_param_logloss['logloss'], "roc auc score": self.best_xgb_param_logloss['auc'],
                      "MSE": self.best_xgb_param_logloss['MSE'], "accuracy": self.best_xgb_param_logloss['accuracy'],
                      "pr auc score": self.best_xgb_param_logloss['pr_auc_score']}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            df_row = {"model": "XGBoost Test set", "year": train_years[-1] + 3, "f1_score": f1_scores_xgb_test_year,
                      "precision": precision, "recall": recall,
                      "logloss": logloss_xgb_test_year, "roc auc score": auc_xgb_test_year,
                      "MSE": MSE_xgb_test_year, "accuracy": accuracy_xgb_test_year,
                      "pr auc score": pr_auc_xgb_test_year}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            self.df_metric_yearly_results.to_csv("metric_yearly_results/df_metric_yearly_results_xgb.csv", index=False)

            if train_years[-1] == 2019: #here
                break
            else:
                train_years.append(train_years[-1] + 1)
                count += 1

        all_data = df_bestk.to_csv("df_bestk/df_bestk_XGB.csv", index=False)

        all_data = self.df_final_result.to_csv("df_results/df_results_XGB.csv", index=False)

        f1_scores_xgb_train = f1_scores_xgb_train / count
        precision_xgb_train = precision_xgb_train / count
        recall_xgb_train = recall_xgb_train / count
        auc_xgb_train = auc_xgb_train / count
        logloss_xgb_train = logloss_xgb_train / count
        MSE_xgb_train = MSE_xgb_train / count
        accuracy_xgb_train = accuracy_xgb_train / count
        pr_auc_xgb_train = pr_auc_xgb_train / count

        f1_scores_xgb_valid = f1_scores_xgb_valid / count
        precision_xgb_valid = precision_xgb_valid / count
        recall_xgb_valid = recall_xgb_valid / count
        auc_xgb_valid = auc_xgb_valid / count
        logloss_xgb_valid = logloss_xgb_valid / count
        MSE_xgb_valid = MSE_xgb_valid / count
        accuracy_xgb_valid = accuracy_xgb_valid / count
        pr_auc_xgb_valid = pr_auc_xgb_valid / count

        f1_scores_xgb_test = f1_scores_xgb_test / count
        precision_xgb_test = precision_xgb_test / count
        recall_xgb_test = recall_xgb_test / count
        auc_xgb_test = auc_xgb_test / count
        logloss_xgb_test = logloss_xgb_test / count
        MSE_xgb_test = MSE_xgb_test / count
        accuracy_xgb_test = accuracy_xgb_test / count
        pr_auc_xgb_test = pr_auc_xgb_test / count

        # plot the precision-recall curves
        pyplot.plot(xgb_recall, xgb_precision, marker='.', label='XGBoost')

        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()

        end = time.time()
        print(end - start)
        print('done')

        df_row = {"model": "XGBoost Training set", "f1_score": f1_scores_xgb_train,
                  "precision": precision_xgb_train, "recall": recall_xgb_train,
                  "logloss": logloss_xgb_train, "roc auc score": auc_xgb_train,
                  "MSE": MSE_xgb_train, "accuracy": accuracy_xgb_train,
                  "pr auc score": pr_auc_xgb_train}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        df_row = {"model": "XGBoost Validation set", "f1_score": f1_scores_xgb_valid,
                  "precision": precision_xgb_valid, "recall": recall_xgb_valid,
                  "logloss": logloss_xgb_valid, "roc auc score": auc_xgb_valid,
                  "MSE": MSE_xgb_valid, "accuracy": accuracy_xgb_valid,
                  "pr auc score": pr_auc_xgb_valid}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        df_row = {"model": "XGBoost Test set", "f1_score": f1_scores_xgb_test,
                  "precision": precision_xgb_test, "recall": recall_xgb_test,
                  "logloss": logloss_xgb_test, "roc auc score": auc_xgb_test,
                  "MSE": MSE_xgb_test, "accuracy": accuracy_xgb_test,
                  "pr auc score": pr_auc_xgb_test}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        self.df_metric_results.to_csv("metric_results/df_metric_results_xgb.csv", index=False)
        

class CatBoost(Model):
    def __init__(self):
        Model.__init__(self)

        self.best_cgb_param_logloss = {
            'boosting_rounds': 0,
            'max_depth': 0,
            'learning_rate': 0,
            'reg_lambda': 0,
            'subsample': 0,
            'colsample_bylevel': 0,
            'min_data_in_leaf': 0,  # gamma equivalent
            'f1_score': 0,
            'precision': 0,
            'recall': 0,
            'logloss': 0,
            'auc': 0,
            'MSE': 0,
            'accuracy': 0,
            'pr_auc_score': 0,
            'f1_score_train': 0,
            'precision_train': 0,
            'recall_train': 0,
            'logloss_train': 0,
            'auc_train': 0,
            'MSE_train': 0,
            'accuracy_train': 0,
            'pr_auc_score_train': 0,
            'kneighbour': 0
        }
    
    def run(self):
        start = time.time()

        boosting_round_cgb = [100,200,300]
        max_depth_cgb = [6]
        learning_rate_cgb = [0.1,0.5,0.9]
        reg_lambda_cgb = [7]
        subsample_cgb = [1]
        colsample_bylevel_cgb = [0.5]
        min_data_in_leaf_cgb = [12]

        auc_cgb_train, logloss_cgb_train, f1_scores_cgb_train, MSE_cgb_train, accuracy_cgb_train, precision_cgb_train, recall_cgb_train, pr_auc_cgb_train = 0, 0, 0, 0, 0, 0, 0, 0
        auc_cgb_valid, logloss_cgb_valid, f1_scores_cgb_valid, MSE_cgb_valid, accuracy_cgb_valid, precision_cgb_valid, recall_cgb_valid, pr_auc_cgb_valid = 0, 0, 0, 0, 0, 0, 0, 0
        auc_cgb_test, logloss_cgb_test, f1_scores_cgb_test, MSE_cgb_test, accuracy_cgb_test, precision_cgb_test, recall_cgb_test, pr_auc_cgb_test = 0, 0, 0, 0, 0, 0, 0, 0

        filename = 'saved_models/cgb_model.sav'

        train_years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
        count = 1

        #numofneighbours = [1, 3, 5, 7, 9, 11]
        numofneighbours = [1,3,5,7,9,11]

        df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
        df_bestk = pd.DataFrame([], columns=["yearmodel", "bestk"])

        for year in train_years:
            auc_cgb_valid_year, logloss_cgb_valid_year, f1_scores_cgb_valid_year, MSE_cgb_valid_year, accuracy_cgb_valid_year, \
            precision_cgb_valid_year, recall_cgb_valid_year, pr_auc_cgb_valid_year = 0, 0, 0, 0, 0, 0, 0, 0

            auc_cgb_train_year, logloss_cgb_train_year, f1_scores_cgb_train_year, MSE_cgb_train_year, accuracy_cgb_train_year, \
            precision_cgb_train_year, recall_cgb_train_year, pr_auc_cgb_train_year = 0, 0, 0, 0, 0, 0, 0, 0
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
                                              "age", "career year","rider_bmi", "win_ratio_slope",
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
                                              "in_top20_next_year"])

            for k in numofneighbours:
                df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
                df_imputed = KNNAlter(self.df_all, df_imputed, k)
                # all_data = df_imputed.to_csv("df_imputed.csv", index=False)

                df_train = df_imputed.loc[df_imputed['year'].isin(train_years)]
                df_valid = df_imputed.loc[df_imputed['year'] == train_years[-1] + 1]
                df_test = df_imputed.loc[df_imputed['year'] == train_years[-1] + 2]
                X_train, y_train = df_train.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_train[
                                       'in_top20_next_year'].copy()
                X_valid, y_valid = df_valid.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_valid[
                                       'in_top20_next_year'].copy()
                X_test, y_test = df_test.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_test[
                                     'in_top20_next_year'].copy()

                # insert previous UPsampling SMOTE code here if needed
                oversample = SMOTE()
                X_train, y_train = oversample.fit_resample(X_train, y_train)

                # feature_selection
                feature_names = list(X_train.columns.values)
                selector = SelectKBest(score_func=mutual_info_classif, k=30)
                X_train = selector.fit_transform(X_train, y_train)  # k optimal?
                mask = selector.get_support()  # list of booleans
                new_features = []
                removed_features = []
                for bool, feature in zip(mask, feature_names):
                    if bool:
                        new_features.append(feature)
                    else:
                        removed_features.append(feature)
                X_train = pd.DataFrame(X_train, columns=new_features)

                # remove all feature in xvalid and xtest that are not in xtrain anymore
                X_valid = X_valid.drop(columns=removed_features)
                X_test = X_test.drop(columns=removed_features)

                for max_depth in max_depth_cgb:
                    for subsample in subsample_cgb:
                        for lr in learning_rate_cgb:
                            for rounds in boosting_round_cgb:
                                for reg_lambda in reg_lambda_cgb:
                                    for colsample in colsample_bylevel_cgb:
                                        for min_data in min_data_in_leaf_cgb:
                                            cb_clf = cgb.CatBoostClassifier(n_estimators=rounds,
                                                                            max_depth=max_depth,
                                                                            subsample=subsample, learning_rate=lr,
                                                                            l2_leaf_reg=reg_lambda,
                                                                            colsample_bylevel=colsample,
                                                                            min_data_in_leaf=min_data,
                                                                            silent=True,
                                                                            loss_function='CrossEntropy',
                                                                            eval_metric='F1')
                                            cb_clf.fit(X_train, y_train)

                                            y_pred = cb_clf.predict(X_valid)
                                            y_pred_train = cb_clf.predict(X_train)

                                            if (metrics.f1_score(y_valid,
                                                                 y_pred) > f1_scores_cgb_valid_year) or f1_scores_cgb_valid_year == 0:
                                                logloss_cgb_valid_year = metrics.log_loss(y_valid, y_pred)
                                                fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred)
                                                auc_cgb_valid_year = metrics.auc(fpr, tpr)
                                                f1_scores_cgb_valid_year = metrics.f1_score(y_valid, y_pred)
                                                MSE_cgb_valid_year = metrics.mean_squared_error(y_valid, y_pred)
                                                accuracy_cgb_valid_year = metrics.accuracy_score(y_valid, y_pred)
                                                precision_cgb_valid_year, recall_cgb_valid_year, fscore, support = metrics.precision_recall_fscore_support(
                                                    y_valid, y_pred, average='weighted')
                                                cgb_probs = cb_clf.predict_proba(X_valid)
                                                # keep probabilities for the positive outcome only
                                                cgb_probs = cgb_probs[:, 1]
                                                cgb_precision, cgb_recall, _ = metrics.precision_recall_curve(
                                                    y_valid, cgb_probs)
                                                pr_auc_cgb_valid_year = metrics.auc(cgb_recall, cgb_precision)

                                                logloss_cgb_train_year = metrics.log_loss(y_train, y_pred_train)
                                                fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_train)
                                                auc_cgb_train_year = metrics.auc(fpr, tpr)
                                                f1_scores_cgb_train_year = metrics.f1_score(y_train, y_pred_train)
                                                MSE_cgb_train_year = metrics.mean_squared_error(y_train,
                                                                                                y_pred_train)
                                                accuracy_cgb_train_year = metrics.accuracy_score(y_train,
                                                                                                 y_pred_train)
                                                precision_cgb_train_year, recall_cgb_train_year, fscore, support = metrics.precision_recall_fscore_support(
                                                    y_train, y_pred_train, average='weighted')
                                                cgb_probs = cb_clf.predict_proba(X_train)
                                                # keep probabilities for the positive outcome only
                                                cgb_probs = cgb_probs[:, 1]
                                                cgb_precision, cgb_recall, _ = metrics.precision_recall_curve(
                                                    y_train, cgb_probs)
                                                pr_auc_cgb_train_year = metrics.auc(cgb_recall, cgb_precision)

                                                self.best_cgb_param_logloss = {
                                                    'max_depth': max_depth,
                                                    'learning_rate': lr,
                                                    'boosting_rounds': rounds,
                                                    'subsample': subsample,
                                                    'reg_lambda': reg_lambda,
                                                    'colsample_bylevel': colsample,
                                                    'min_data_in_leaf': min_data,
                                                    'logloss': logloss_cgb_valid_year,
                                                    'f1_score': f1_scores_cgb_valid_year,
                                                    'MSE': MSE_cgb_valid_year,
                                                    'auc': auc_cgb_valid_year,
                                                    'accuracy': accuracy_cgb_valid_year,
                                                    'precision': precision_cgb_valid_year,
                                                    'recall': recall_cgb_valid_year,
                                                    'pr_auc_score': pr_auc_cgb_valid_year,
                                                    'logloss_train': logloss_cgb_train_year,
                                                    'auc_train': auc_cgb_train_year,
                                                    'f1_score_train': f1_scores_cgb_train_year,
                                                    'MSE_train': MSE_cgb_train_year,
                                                    'accuracy_train': accuracy_cgb_train_year,
                                                    'precision_train': precision_cgb_train_year,
                                                    'recall_train': recall_cgb_train_year,
                                                    'pr_auc_score_train': pr_auc_cgb_train_year,
                                                    'kneighbour': k
                                                }

            df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
            df_imputed = KNNAlter(self.df_all, df_imputed, self.best_cgb_param_logloss["kneighbour"])
            all_data = df_imputed.to_csv("df_imputed/df_imputed_CGB.csv", index=False)

            df_train = df_imputed.loc[df_imputed['year'].isin(train_years)]
            df_valid = df_imputed.loc[df_imputed['year'] == train_years[-1] + 1]
            df_test = df_imputed.loc[df_imputed['year'] == train_years[-1] + 2]
            X_train, y_train = df_train.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_train[
                                   'in_top20_next_year'].copy()
            X_valid, y_valid = df_valid.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_valid[
                                   'in_top20_next_year'].copy()
            X_test, y_test = df_test.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_test[
                                 'in_top20_next_year'].copy()

            # insert previous UPsampling SMOTE code here if needed
            oversample = SMOTE()
            X_train, y_train = oversample.fit_resample(X_train, y_train)

            # feature_selection
            feature_names = list(X_train.columns.values)
            selector = SelectKBest(score_func=mutual_info_classif, k=30)
            X_train = selector.fit_transform(X_train, y_train)  # k optimal?
            mask = selector.get_support()  # list of booleans
            new_features = []
            removed_features = []
            for bool, feature in zip(mask, feature_names):
                if bool:
                    new_features.append(feature)
                else:
                    removed_features.append(feature)
            X_train = pd.DataFrame(X_train, columns=new_features)

            # remove all feautre in xvalid and xtest that are not in xtrain anymore
            X_valid = X_valid.drop(columns=removed_features)
            X_test = X_test.drop(columns=removed_features)

            frames = [X_train, X_valid]
            xtv = pd.concat(frames)
            frames = [y_train, y_valid]
            ytv = pd.concat(frames)
            cb_clf = cgb.CatBoostClassifier(n_estimators=self.best_cgb_param_logloss['boosting_rounds'],
                                            max_depth=self.best_cgb_param_logloss['max_depth'],
                                            subsample=self.best_cgb_param_logloss['subsample'],
                                            learning_rate=self.best_cgb_param_logloss['learning_rate'],
                                            l2_leaf_reg=self.best_cgb_param_logloss['reg_lambda'],
                                            colsample_bylevel=self.best_cgb_param_logloss['colsample_bylevel'],
                                            min_data_in_leaf=self.best_cgb_param_logloss['min_data_in_leaf'],
                                            silent=True, loss_function='CrossEntropy', eval_metric='F1')
            cb_clf.fit(xtv, ytv)

            pickle.dump(cb_clf, open(filename, 'wb'))
            y_pred = cb_clf.predict(X_test)

            # prepare df beforehand and then compare dfall and xtest,ypred and append to df and then write to csv
            # row and y_pred to csv with their names included so gotta compare with df_imputed
            # iterate through dataframe:
            for index, row in X_test.iterrows():
                df_row = df_imputed.loc[
                    (df_imputed[new_features[0]] == row[new_features[0]])
                    & (df_imputed[new_features[1]] == row[
                        new_features[1]])
                    & (df_imputed[new_features[2]] == row[new_features[2]]) & (
                            df_imputed[new_features[3]] == row[new_features[3]])
                    & (df_imputed[new_features[4]] == row[new_features[4]]) & (
                            df_imputed[new_features[5]] == row[new_features[5]])
                    ]

                df_result = df_result.append(df_row, ignore_index=True)
            pred = pd.Series(y_pred, name='Prediction')

            frames = [df_result, pred]
            df_result = pd.concat(frames, axis=1)
            self.df_final_result = self.df_final_result.append(df_result)

            auc_cgb_valid += self.best_cgb_param_logloss['auc']
            logloss_cgb_valid += self.best_cgb_param_logloss['logloss']
            f1_scores_cgb_valid += self.best_cgb_param_logloss['f1_score']
            MSE_cgb_valid += self.best_cgb_param_logloss['MSE']
            accuracy_cgb_valid += self.best_cgb_param_logloss['accuracy']
            precision_cgb_valid += self.best_cgb_param_logloss['precision']
            recall_cgb_valid += self.best_cgb_param_logloss['recall']
            pr_auc_cgb_valid += self.best_cgb_param_logloss['pr_auc_score']

            auc_cgb_train += self.best_cgb_param_logloss['auc_train']
            logloss_cgb_train += self.best_cgb_param_logloss['logloss_train']
            f1_scores_cgb_train += self.best_cgb_param_logloss['f1_score_train']
            MSE_cgb_train += self.best_cgb_param_logloss['MSE_train']
            accuracy_cgb_train += self.best_cgb_param_logloss['accuracy_train']
            precision_cgb_train += self.best_cgb_param_logloss['precision_train']
            recall_cgb_train += self.best_cgb_param_logloss['recall_train']
            pr_auc_cgb_train += self.best_cgb_param_logloss['pr_auc_score_train']

            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            auc_cgb_test_year = metrics.auc(fpr, tpr)
            logloss_cgb_test_year = metrics.log_loss(y_test, y_pred)
            f1_scores_cgb_test_year = metrics.f1_score(y_test, y_pred)
            MSE_cgb_test_year = metrics.mean_squared_error(y_test, y_pred)
            accuracy_cgb_test_year = metrics.accuracy_score(y_test, y_pred)
            precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, y_pred,
                                                                                         average='weighted')

            auc_cgb_test += auc_cgb_test_year
            logloss_cgb_test += logloss_cgb_test_year
            f1_scores_cgb_test += f1_scores_cgb_test_year
            MSE_cgb_test += MSE_cgb_test_year
            accuracy_cgb_test += accuracy_cgb_test_year
            precision_cgb_test = precision + precision_cgb_test
            recall_cgb_test = recall + recall_cgb_test

            cgb_probs = cb_clf.predict_proba(X_test)
            # keep probabilities for the positive outcome only
            cgb_probs = cgb_probs[:, 1]
            # predict class values
            y_pred = cb_clf.predict(X_test)
            cgb_precision, cgb_recall, _ = metrics.precision_recall_curve(y_test, cgb_probs)
            pr_auc_cgb_test_year = metrics.auc(cgb_recall, cgb_precision)
            pr_auc_cgb_test += pr_auc_cgb_test_year

            df_row = {"yearmodel": train_years[-1] + 3, "bestk": self.best_cgb_param_logloss["kneighbour"]}
            df_bestk = df_bestk.append(df_row, ignore_index=True)

            df_row = {"model": "CatBoost Training set", "year": train_years[-1] + 3,
                      "f1_score": self.best_cgb_param_logloss['f1_score_train'],
                      "precision": self.best_cgb_param_logloss['precision_train'],
                      "recall": self.best_cgb_param_logloss['recall_train'],
                      "logloss": self.best_cgb_param_logloss['logloss_train'],
                      "roc auc score": self.best_cgb_param_logloss['auc_train'],
                      "MSE": self.best_cgb_param_logloss['MSE_train'],
                      "accuracy": self.best_cgb_param_logloss['accuracy_train'],
                      "pr auc score": self.best_cgb_param_logloss['pr_auc_score_train']}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            df_row = {"model": "CatBoost Validation set", "year": train_years[-1] + 3,
                      "f1_score": self.best_cgb_param_logloss['f1_score'],
                      "precision": self.best_cgb_param_logloss['precision'],
                      "recall": self.best_cgb_param_logloss['recall'],
                      "logloss": self.best_cgb_param_logloss['logloss'],
                      "roc auc score": self.best_cgb_param_logloss['auc'],
                      "MSE": self.best_cgb_param_logloss['MSE'],
                      "accuracy": self.best_cgb_param_logloss['accuracy'],
                      "pr auc score": self.best_cgb_param_logloss['pr_auc_score']}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            df_row = {"model": "CatBoost Test set", "year": train_years[-1] + 3,
                      "f1_score": f1_scores_cgb_test_year,
                      "precision": precision, "recall": recall,
                      "logloss": logloss_cgb_test_year, "roc auc score": auc_cgb_test_year,
                      "MSE": MSE_cgb_test_year,
                      "accuracy": accuracy_cgb_test_year,
                      "pr auc score": pr_auc_cgb_test_year}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            self.df_metric_yearly_results.to_csv("metric_yearly_results/df_metric_yearly_results_cgb.csv", index=False)

            if train_years[-1] == 2019: #here
                break
            else:
                train_years.append(train_years[-1] + 1)
                count += 1

        all_data = df_bestk.to_csv("df_bestk/df_bestk_CGB.csv", index=False)

        all_data = self.df_final_result.to_csv("df_results/df_results_CGB.csv", index=False)

        auc_cgb_train = auc_cgb_train / count
        logloss_cgb_train = logloss_cgb_train / count
        f1_scores_cgb_train = f1_scores_cgb_train / count
        MSE_cgb_train = MSE_cgb_train / count
        accuracy_cgb_train = accuracy_cgb_train / count
        precision_cgb_train = precision_cgb_train / count
        recall_cgb_train = recall_cgb_train / count
        pr_auc_cgb_train = pr_auc_cgb_train / count

        auc_cgb_valid = auc_cgb_valid / count
        logloss_cgb_valid = logloss_cgb_valid / count
        f1_scores_cgb_valid = f1_scores_cgb_valid / count
        MSE_cgb_valid = MSE_cgb_valid / count
        accuracy_cgb_valid = accuracy_cgb_valid / count
        precision_cgb_valid = precision_cgb_valid / count
        recall_cgb_valid = recall_cgb_valid / count
        pr_auc_cgb_valid = pr_auc_cgb_valid / count

        auc_cgb_test = auc_cgb_test / count
        logloss_cgb_test = logloss_cgb_test / count
        f1_scores_cgb_test = f1_scores_cgb_test / count
        MSE_cgb_test = MSE_cgb_test / count
        accuracy_cgb_test = accuracy_cgb_test / count
        precision_cgb_test = precision_cgb_test / count
        recall_cgb_test = recall_cgb_test / count
        pr_auc_cgb_test = pr_auc_cgb_test / count

        # plot the precision-recall curves
        pyplot.plot(cgb_recall, cgb_precision, marker='.', label='CatBoost')

        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()

        end = time.time()
        print(end - start)
        print('done')

        df_row = {"model": "CatBoost Training set", "f1_score": f1_scores_cgb_train,
                  "precision": precision_cgb_train, "recall": recall_cgb_train,
                  "logloss": logloss_cgb_train, "roc auc score": auc_cgb_train,
                  "MSE": MSE_cgb_train, "accuracy": accuracy_cgb_train,
                  "pr auc score": pr_auc_cgb_train}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        df_row = {"model": "CatBoost Validation set", "f1_score": f1_scores_cgb_valid,
                  "precision": precision_cgb_valid, "recall": recall_cgb_valid,
                  "logloss": logloss_cgb_valid, "roc auc score": auc_cgb_valid,
                  "MSE": MSE_cgb_valid, "accuracy": accuracy_cgb_valid,
                  "pr auc score": pr_auc_cgb_valid}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        df_row = {"model": "CatBoost Test set", "f1_score": f1_scores_cgb_test,
                  "precision": precision_cgb_test, "recall": recall_cgb_test,
                  "logloss": logloss_cgb_test, "roc auc score": auc_cgb_test,
                  "MSE": MSE_cgb_test, "accuracy": accuracy_cgb_test,
                  "pr auc score": pr_auc_cgb_test}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        self.df_metric_results.to_csv("metric_results/df_metric_results_cgb.csv", index=False)

class LightGBM(Model):
    def __init__(self):
        Model.__init__(self)
        self.best_lgb_param_logloss = {
            'reg_alpha': 0,  # or lambda_l1
            'boosting_rounds': 0,  # n_estimators
            'max_depth': 0,
            'learning_rate': 0,
            'reg_lambda': 0,
            'subsample': 0,
            'colsample_bytree': 0,
            'min_split_gain': 0,  # gamma
            'num_leaves': 0,
            'min_data_in_leaf': 0,
            'logloss': 0,
            'auc': 0,
            'f1_score': 0,
            'MSE': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'pr_auc_score': 0,
            'logloss_train': 0,
            'auc_train': 0,
            'f1_score_train': 0,
            'MSE_train': 0,
            'accuracy_train': 0,
            'precision_train': 0,
            'recall_train': 0,
            'pr_auc_score_train': 0,
            'kneighbour': 0
        }

    def run(self):
        start = time.time()

        alpha_lgb = [4]
        boosting_round_lgb = [100]
        max_depth_lgb = [6]
        learning_rate_lgb = [0.1]
        reg_lambda_lgb = [1]
        subsample_lgb = [0.5]
        colsample_bytree_lgb = [0.1]
        min_split_gain_lgb = [4]
        num_leaves_lgb = [30]
        min_data_in_leaf_lgb = [15]

        auc_lgb_train, logloss_lgb_train, f1_scores_lgb_train, MSE_lgb_train, accuracy_lgb_train, precision_lgb_train, recall_lgb_train, pr_auc_lgb_train = 0, 0, 0, 0, 0, 0, 0, 0
        auc_lgb_valid, logloss_lgb_valid, f1_scores_lgb_valid, MSE_lgb_valid, accuracy_lgb_valid, precision_lgb_valid, recall_lgb_valid, pr_auc_lgb_valid = 0, 0, 0, 0, 0, 0, 0, 0
        auc_lgb_test, logloss_lgb_test, f1_scores_lgb_test, MSE_lgb_test, accuracy_lgb_test, precision_lgb_test, recall_lgb_test, pr_auc_lgb_test = 0, 0, 0, 0, 0, 0, 0, 0

        filename = 'saved_models/lgb_model.sav'

        train_years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
        # train_years = [2009]
        count = 1
        # lgb_recall_all = np.array([])
        # lgb_precision_all = np.array([])

        #numofneighbours = [1, 3, 5, 7, 9, 11]
        numofneighbours = [1]

        df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
        df_bestk = pd.DataFrame([], columns=["yearmodel", "bestk"])

        for year in train_years:
            auc_lgb_valid_year, logloss_lgb_valid_year, f1_scores_lgb_valid_year, MSE_lgb_valid_year, accuracy_lgb_valid_year, \
            precision_lgb_valid_year, recall_lgb_valid_year, pr_auc_lgb_valid_year = 0, 0, 0, 0, 0, 0, 0, 0

            auc_lgb_train_year, logloss_lgb_train_year, f1_scores_lgb_train_year, MSE_lgb_train_year, accuracy_lgb_train_year, \
            precision_lgb_train_year, recall_lgb_train_year, pr_auc_lgb_train_year = 0, 0, 0, 0, 0, 0, 0, 0
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
                                              "age", "career year","rider_bmi", "win_ratio_slope",
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
                                              "in_top20_next_year"])

            for k in numofneighbours:
                df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
                df_imputed = KNNAlter(self.df_all, df_imputed, k)
                # all_data = df_imputed.to_csv("df_imputed.csv", index=False)

                df_train = df_imputed.loc[df_imputed['year'].isin(train_years)]
                df_valid = df_imputed.loc[df_imputed['year'] == train_years[-1] + 1]
                df_test = df_imputed.loc[df_imputed['year'] == train_years[-1] + 2]
                X_train, y_train = df_train.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_train[
                                       'in_top20_next_year'].copy()
                X_valid, y_valid = df_valid.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_valid[
                                       'in_top20_next_year'].copy()
                X_test, y_test = df_test.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_test[
                                     'in_top20_next_year'].copy()

                # insert previous UPsampling SMOTE code here if needed
                oversample = SMOTE()
                X_train, y_train = oversample.fit_resample(X_train, y_train)

                # feature_selection
                feature_names = list(X_train.columns.values)
                selector = SelectKBest(score_func=mutual_info_classif, k=30)
                X_train = selector.fit_transform(X_train, y_train)  # k optimal?
                mask = selector.get_support()  # list of booleans
                new_features = []
                removed_features = []
                for bool, feature in zip(mask, feature_names):
                    if bool:
                        new_features.append(feature)
                    else:
                        removed_features.append(feature)
                X_train = pd.DataFrame(X_train, columns=new_features)

                # remove all feature in xvalid and xtest that are not in xtrain anymore
                X_valid = X_valid.drop(columns=removed_features)
                X_test = X_test.drop(columns=removed_features)

                for max_depth in max_depth_lgb:
                    for num_leaves in num_leaves_lgb:
                        for min_data in min_data_in_leaf_lgb:
                            for colsample in colsample_bytree_lgb:
                                for subsample in subsample_lgb:
                                    for lr in learning_rate_lgb:
                                        for alpha in alpha_lgb:
                                            for round in boosting_round_lgb:
                                                for landa in reg_lambda_lgb:
                                                    for split_gain in min_split_gain_lgb:
                                                        lgb_clf = LGBMClassifier(max_depth=max_depth,
                                                                                 num_leaves=num_leaves, reg_alpha=alpha,
                                                                                 n_estimators=round, reg_lambda=landa,
                                                                                 min_split_gain=split_gain,
                                                                                 min_data_in_leaf=min_data,
                                                                                 colsample_bytree=colsample,
                                                                                 subsample=subsample,
                                                                                 learning_rate=lr)
                                                        lgb_clf.fit(X_train, y_train)
                                                        y_pred = lgb_clf.predict(X_valid)
                                                        y_pred_train = lgb_clf.predict(X_train)

                                                        if (metrics.f1_score(y_valid,
                                                                             y_pred) > f1_scores_lgb_valid_year) or f1_scores_lgb_valid_year == 0:
                                                            logloss_lgb_valid_year = metrics.log_loss(y_valid, y_pred)
                                                            fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred)
                                                            auc_lgb_valid_year = metrics.auc(fpr, tpr)
                                                            f1_scores_lgb_valid_year = metrics.f1_score(y_valid, y_pred)
                                                            MSE_lgb_valid_year = metrics.mean_squared_error(y_valid,
                                                                                                            y_pred)
                                                            accuracy_lgb_valid_year = metrics.accuracy_score(y_valid,
                                                                                                             y_pred)
                                                            precision_lgb_valid_year, recall_lgb_valid_year, fscore, support = metrics.precision_recall_fscore_support(
                                                                y_valid, y_pred, average='weighted')
                                                            lgb_probs = lgb_clf.predict_proba(X_valid)
                                                            # keep probabilities for the positive outcome only
                                                            lgb_probs = lgb_probs[:, 1]
                                                            lgb_precision, lgb_recall, _ = metrics.precision_recall_curve(
                                                                y_valid, lgb_probs)
                                                            pr_auc_lgb_valid_year = metrics.auc(lgb_recall,
                                                                                                lgb_precision)

                                                            logloss_lgb_train_year = metrics.log_loss(y_train,
                                                                                                      y_pred_train)
                                                            fpr, tpr, thresholds = metrics.roc_curve(y_train,
                                                                                                     y_pred_train)
                                                            auc_lgb_train_year = metrics.auc(fpr, tpr)
                                                            f1_scores_lgb_train_year = metrics.f1_score(y_train,
                                                                                                        y_pred_train)
                                                            MSE_lgb_train_year = metrics.mean_squared_error(y_train,
                                                                                                            y_pred_train)
                                                            accuracy_lgb_train_year = metrics.accuracy_score(y_train,
                                                                                                             y_pred_train)
                                                            precision_lgb_train_year, recall_lgb_train_year, fscore, support = metrics.precision_recall_fscore_support(
                                                                y_train, y_pred_train, average='weighted')
                                                            lgb_probs = lgb_clf.predict_proba(X_train)
                                                            # keep probabilities for the positive outcome only
                                                            lgb_probs = lgb_probs[:, 1]
                                                            lgb_precision, lgb_recall, _ = metrics.precision_recall_curve(
                                                                y_train, lgb_probs)
                                                            pr_auc_lgb_train_year = metrics.auc(lgb_recall,
                                                                                                lgb_precision)

                                                            lgb = LGBMClassifier(max_depth=max_depth,
                                                                                 num_leaves=num_leaves, reg_alpha=alpha,
                                                                                 n_estimators=round, reg_lambda=landa,
                                                                                 min_split_gain=split_gain,
                                                                                 min_data_in_leaf=min_data,
                                                                                 colsample_bytree=colsample,
                                                                                 subsample=subsample, learning_rate=lr)

                                                            self.best_lgb_param_logloss = {
                                                                'reg_alpha': alpha,
                                                                'max_depth': max_depth,
                                                                'boosting_rounds': round,
                                                                'reg_lambda': landa,
                                                                'min_split_gain': split_gain,
                                                                'num_leaves': num_leaves,
                                                                'min_data_in_leaf': min_data,
                                                                'colsample_bytree': colsample,
                                                                'subsample': subsample,
                                                                'learning_rate': lr,
                                                                'logloss': logloss_lgb_valid_year,
                                                                'f1_score': f1_scores_lgb_valid_year,
                                                                'MSE': MSE_lgb_valid_year,
                                                                'auc': auc_lgb_valid_year,
                                                                'accuracy': accuracy_lgb_valid_year,
                                                                'precision': precision_lgb_valid_year,
                                                                'recall': recall_lgb_valid_year,
                                                                'pr_auc_score': pr_auc_lgb_valid_year,
                                                                'logloss_train': logloss_lgb_train_year,
                                                                'auc_train': auc_lgb_train_year,
                                                                'f1_score_train': f1_scores_lgb_train_year,
                                                                'MSE_train': MSE_lgb_train_year,
                                                                'accuracy_train': accuracy_lgb_train_year,
                                                                'precision_train': precision_lgb_train_year,
                                                                'recall_train': recall_lgb_train_year,
                                                                'pr_auc_score_train': pr_auc_lgb_train_year,
                                                                'kneighbour': k
                                                            }

            df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
            df_imputed = KNNAlter(self.df_all, df_imputed, self.best_lgb_param_logloss['kneighbour'])
            all_data = df_imputed.to_csv("df_imputed/df_imputed_LGB.csv", index=False)

            df_train = df_imputed.loc[df_imputed['year'].isin(train_years)]
            df_valid = df_imputed.loc[df_imputed['year'] == train_years[-1] + 1]
            df_test = df_imputed.loc[df_imputed['year'] == train_years[-1] + 2]
            X_train, y_train = df_train.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_train[
                                   'in_top20_next_year'].copy()
            X_valid, y_valid = df_valid.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_valid[
                                   'in_top20_next_year'].copy()
            X_test, y_test = df_test.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_test[
                                 'in_top20_next_year'].copy()

            # insert previous UPsampling SMOTE code here if needed
            oversample = SMOTE()
            X_train, y_train = oversample.fit_resample(X_train, y_train)

            # feature_selection
            feature_names = list(X_train.columns.values)
            selector = SelectKBest(score_func=mutual_info_classif, k=30) #k optimal?
            X_train = selector.fit_transform(X_train, y_train)
            mask = selector.get_support()  # list of booleans
            new_features = []
            removed_features = []
            for bool, feature in zip(mask, feature_names):
                if bool:
                    new_features.append(feature)
                else:
                    removed_features.append(feature)
            X_train = pd.DataFrame(X_train, columns=new_features)

            # remove all feautre in xvalid and xtest that are not in xtrain anymore
            X_valid = X_valid.drop(columns=removed_features)
            X_test = X_test.drop(columns=removed_features)

            frames = [X_train, X_valid]
            xtv = pd.concat(frames)
            frames = [y_train, y_valid]
            ytv = pd.concat(frames)
            lgb_clf = LGBMClassifier(max_depth=self.best_lgb_param_logloss['max_depth'],
                                     num_leaves=self.best_lgb_param_logloss['num_leaves'],
                                     reg_alpha=self.best_lgb_param_logloss['reg_alpha'],
                                     n_estimators=self.best_lgb_param_logloss['boosting_rounds'],
                                     reg_lambda=self.best_lgb_param_logloss['reg_lambda'],
                                     min_split_gain=self.best_lgb_param_logloss['min_split_gain'],
                                     min_data_in_leaf=self.best_lgb_param_logloss['min_data_in_leaf'],
                                     colsample_bytree=self.best_lgb_param_logloss['colsample_bytree'],
                                     subsample=self.best_lgb_param_logloss['subsample'],
                                     learning_rate=self.best_lgb_param_logloss['learning_rate'])
            lgb_clf.fit(xtv, ytv)

            pickle.dump(lgb_clf, open(filename, 'wb'))
            y_pred = lgb_clf.predict(X_test)

            # prepare df beforehand and then compare dfall and xtest,ypred and append to df and then write to csv
            # row and y_pred to csv with their names included so gotta compare with df_imputed
            # iterate through dataframe:
            for index, row in X_test.iterrows():
                df_row = df_imputed.loc[
                    (df_imputed[new_features[0]] == row[new_features[0]])
                    & (df_imputed[new_features[1]] == row[
                        new_features[1]])
                    & (df_imputed[new_features[2]] == row[new_features[2]]) & (
                            df_imputed[new_features[3]] == row[new_features[3]])
                    & (df_imputed[new_features[4]] == row[new_features[4]]) & (
                            df_imputed[new_features[5]] == row[new_features[5]])
                    ]

                df_result = df_result.append(df_row, ignore_index=True)
            pred = pd.Series(y_pred, name='Prediction')

            frames = [df_result, pred]
            df_result = pd.concat(frames, axis=1)
            self.df_final_result = self.df_final_result.append(df_result)

            auc_lgb_valid += self.best_lgb_param_logloss['auc']
            logloss_lgb_valid += self.best_lgb_param_logloss['logloss']
            f1_scores_lgb_valid += self.best_lgb_param_logloss['f1_score']
            MSE_lgb_valid += self.best_lgb_param_logloss['MSE']
            accuracy_lgb_valid += self.best_lgb_param_logloss['accuracy']
            precision_lgb_valid += self.best_lgb_param_logloss['precision']
            recall_lgb_valid += self.best_lgb_param_logloss['recall']
            pr_auc_lgb_valid += self.best_lgb_param_logloss['pr_auc_score']

            auc_lgb_train += self.best_lgb_param_logloss['auc_train']
            logloss_lgb_train += self.best_lgb_param_logloss['logloss_train']
            f1_scores_lgb_train += self.best_lgb_param_logloss['f1_score_train']
            MSE_lgb_train += self.best_lgb_param_logloss['MSE_train']
            accuracy_lgb_train += self.best_lgb_param_logloss['accuracy_train']
            precision_lgb_train += self.best_lgb_param_logloss['precision_train']
            recall_lgb_train += self.best_lgb_param_logloss['recall_train']
            pr_auc_lgb_train += self.best_lgb_param_logloss['pr_auc_score_train']

            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            auc_lgb_test_year = metrics.auc(fpr, tpr)
            logloss_lgb_test_year = metrics.log_loss(y_test, y_pred)
            f1_scores_lgb_test_year = metrics.f1_score(y_test, y_pred)
            MSE_lgb_test_year = metrics.mean_squared_error(y_test, y_pred)
            accuracy_lgb_test_year = metrics.accuracy_score(y_test, y_pred)
            precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, y_pred,
                                                                                         average='weighted')

            auc_lgb_test += auc_lgb_test_year
            logloss_lgb_test += logloss_lgb_test_year
            f1_scores_lgb_test += f1_scores_lgb_test_year
            MSE_lgb_test += MSE_lgb_test_year
            accuracy_lgb_test += accuracy_lgb_test_year
            precision_lgb_test = precision + precision_lgb_test
            recall_lgb_test = recall + recall_lgb_test

            lgb_probs = lgb_clf.predict_proba(X_test)
            # keep probabilities for the positive outcome only
            lgb_probs = lgb_probs[:, 1]
            # predict class values
            y_pred = lgb_clf.predict(X_test)
            lgb_precision, lgb_recall, _ = metrics.precision_recall_curve(y_test, lgb_probs)
            pr_auc_lgb_test_year = metrics.auc(lgb_recall, lgb_precision)
            pr_auc_lgb_test += pr_auc_lgb_test_year

            df_row = {"yearmodel": train_years[-1] + 3, "bestk": self.best_lgb_param_logloss["kneighbour"]}
            df_bestk = df_bestk.append(df_row, ignore_index=True)

            df_row = {"model": "LightGBM Training set", "year": train_years[-1] + 3,
                      "f1_score": self.best_lgb_param_logloss['f1_score_train'],
                      "precision": self.best_lgb_param_logloss['precision_train'],
                      "recall": self.best_lgb_param_logloss['recall_train'],
                      "logloss": self.best_lgb_param_logloss['logloss_train'],
                      "roc auc score": self.best_lgb_param_logloss['auc_train'],
                      "MSE": self.best_lgb_param_logloss['MSE_train'],
                      "accuracy": self.best_lgb_param_logloss['accuracy_train'],
                      "pr auc score": self.best_lgb_param_logloss['pr_auc_score_train']}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            df_row = {"model": "LightGBM Validation set", "year": train_years[-1] + 3,
                      "f1_score": self.best_lgb_param_logloss['f1_score'],
                      "precision": self.best_lgb_param_logloss['precision'],
                      "recall": self.best_lgb_param_logloss['recall'],
                      "logloss": self.best_lgb_param_logloss['logloss'],
                      "roc auc score": self.best_lgb_param_logloss['auc'],
                      "MSE": self.best_lgb_param_logloss['MSE'],
                      "accuracy": self.best_lgb_param_logloss['accuracy'],
                      "pr auc score": self.best_lgb_param_logloss['pr_auc_score']}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            df_row = {"model": "LightGBM Test set", "year": train_years[-1] + 3,
                      "f1_score": f1_scores_lgb_test_year,
                      "precision": precision, "recall": recall,
                      "logloss": logloss_lgb_test_year, "roc auc score": auc_lgb_test_year,
                      "MSE": MSE_lgb_test_year,
                      "accuracy": accuracy_lgb_test_year,
                      "pr auc score": pr_auc_lgb_test_year}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            self.df_metric_yearly_results.to_csv("metric_yearly_results/df_metric_yearly_results_lgb.csv", index=False)

            if train_years[-1] == 2019: #here
                break
            else:
                train_years.append(train_years[-1] + 1)
                count += 1

        all_data = df_bestk.to_csv("df_bestk/df_bestk_LGB.csv", index=False)

        all_data = self.df_final_result.to_csv("df_results/df_results_LGB.csv", index=False)

        auc_lgb_train = auc_lgb_train / count
        logloss_lgb_train = logloss_lgb_train / count
        f1_scores_lgb_train = f1_scores_lgb_train / count
        MSE_lgb_train = MSE_lgb_train / count
        accuracy_lgb_train = accuracy_lgb_train / count
        precision_lgb_train = precision_lgb_train / count
        recall_lgb_train = recall_lgb_train / count
        pr_auc_lgb_train = pr_auc_lgb_train / count

        auc_lgb_valid = auc_lgb_valid / count
        logloss_lgb_valid = logloss_lgb_valid / count
        f1_scores_lgb_valid = f1_scores_lgb_valid / count
        MSE_lgb_valid = MSE_lgb_valid / count
        accuracy_lgb_valid = accuracy_lgb_valid / count
        precision_lgb_valid = precision_lgb_valid / count
        recall_lgb_valid = recall_lgb_valid / count
        pr_auc_lgb_valid = pr_auc_lgb_valid / count

        auc_lgb_test = auc_lgb_test / count
        logloss_lgb_test = logloss_lgb_test / count
        f1_scores_lgb_test = f1_scores_lgb_test / count
        MSE_lgb_test = MSE_lgb_test / count
        accuracy_lgb_test = accuracy_lgb_test / count
        precision_lgb_test = precision_lgb_test / count
        recall_lgb_test = recall_lgb_test / count
        pr_auc_lgb_test = pr_auc_lgb_test / count

        # plot the precision-recall curves
        pyplot.plot(lgb_recall, lgb_precision, marker='.', label='LightGBM')

        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()

        end = time.time()
        print(end - start)
        print('done')

        df_row = {"model": "LightGBM Training set", "f1_score": f1_scores_lgb_train,
                  "precision": precision_lgb_train, "recall": recall_lgb_train,
                  "logloss": logloss_lgb_train, "roc auc score": auc_lgb_train,
                  "MSE": MSE_lgb_train, "accuracy": accuracy_lgb_train,
                  "pr auc score": pr_auc_lgb_train}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        df_row = {"model": "LightGBM Validation set", "f1_score": f1_scores_lgb_valid,
                  "precision": precision_lgb_valid, "recall": recall_lgb_valid,
                  "logloss": logloss_lgb_valid, "roc auc score": auc_lgb_valid,
                  "MSE": MSE_lgb_valid, "accuracy": accuracy_lgb_valid,
                  "pr auc score": pr_auc_lgb_valid}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        df_row = {"model": "LightGBM Test set", "f1_score": f1_scores_lgb_test,
                  "precision": precision_lgb_test, "recall": recall_lgb_test,
                  "logloss": logloss_lgb_test, "roc auc score": auc_lgb_test,
                  "MSE": MSE_lgb_test, "accuracy": accuracy_lgb_test,
                  "pr auc score": pr_auc_lgb_test}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        self.df_metric_results.to_csv("metric_results/df_metric_results_lgb.csv", index=False)

class RandomForest(Model):
    def __init__(self):
        Model.__init__(self)
        self.best_rf_param_logloss = {
            'n_estimators': 0,
            'max_depth': 0,
            'min_sample_split': 0,
            'logloss': 0,
            'auc': 0,
            'f1_score': 0,
            'MSE': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'pr_auc_score': 0,
            'logloss_train': 0,
            'auc_train': 0,
            'f1_score_train': 0,
            'MSE_train': 0,
            'accuracy_train': 0,
            'precision_train': 0,
            'recall_train': 0,
            'pr_auc_score_train': 0,
            'kneighbour': 0
        }

    def run(self):
        start = time.time()

        n_estimators_rf = [100]
        max_depth_rf = [12]
        min_sample_split_rf = [2]

        auc_rf_train, logloss_rf_train, f1_scores_rf_train, MSE_rf_train, accuracy_rf_train, precision_rf_train, recall_rf_train, pr_auc_rf_train = 0, 0, 0, 0, 0, 0, 0, 0
        auc_rf_valid, logloss_rf_valid, f1_scores_rf_valid, MSE_rf_valid, accuracy_rf_valid, precision_rf_valid, recall_rf_valid, pr_auc_rf_valid = 0, 0, 0, 0, 0, 0, 0, 0
        auc_rf_test, logloss_rf_test, f1_scores_rf_test, MSE_rf_test, accuracy_rf_test, precision_rf_test, recall_rf_test, pr_auc_rf_test = 0, 0, 0, 0, 0, 0, 0, 0

        filename = 'saved_models/rf_model.sav'

        train_years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
        count = 1

        #numofneighbours = [1, 3, 5, 7, 9, 11]
        numofneighbours = [1]

        df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
        df_bestk = pd.DataFrame([], columns=["yearmodel", "bestk"])

        for year in train_years:
            auc_rf_valid_year, logloss_rf_valid_year, f1_scores_rf_valid_year, MSE_rf_valid_year, accuracy_rf_valid_year, \
            precision_rf_valid_year, recall_rf_valid_year, pr_auc_rf_valid_year = 0, 0, 0, 0, 0, 0, 0, 0

            auc_rf_train_year, logloss_rf_train_year, f1_scores_rf_train_year, MSE_rf_train_year, accuracy_rf_train_year, \
            precision_rf_train_year, recall_rf_train_year, pr_auc_rf_train_year = 0, 0, 0, 0, 0, 0, 0, 0
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
                                              "age", "career year","rider_bmi", "win_ratio_slope",
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
                                              "in_top20_next_year"])

            for k in numofneighbours:
                df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
                df_imputed = KNNAlter(self.df_all, df_imputed, k)

                # all_data = df_imputed.to_csv("df_imputed.csv", index=False)

                df_train = df_imputed.loc[df_imputed['year'].isin(train_years)]
                df_valid = df_imputed.loc[df_imputed['year'] == train_years[-1] + 1]
                df_test = df_imputed.loc[df_imputed['year'] == train_years[-1] + 2]
                X_train, y_train = df_train.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_train[
                                       'in_top20_next_year'].copy()
                X_valid, y_valid = df_valid.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_valid[
                                       'in_top20_next_year'].copy()
                X_test, y_test = df_test.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_test[
                                     'in_top20_next_year'].copy()

                # insert previous UPsampling SMOTE code here if needed
                oversample = SMOTE()
                X_train, y_train = oversample.fit_resample(X_train, y_train)

                # feature_selection
                feature_names = list(X_train.columns.values)
                selector = SelectKBest(score_func=mutual_info_classif, k=30)
                X_train = selector.fit_transform(X_train, y_train)  # k optimal?
                mask = selector.get_support()  # list of booleans
                new_features = []
                removed_features = []
                for bool, feature in zip(mask, feature_names):
                    if bool:
                        new_features.append(feature)
                    else:
                        removed_features.append(feature)
                X_train = pd.DataFrame(X_train, columns=new_features)

                # remove all feautre in xvalid and xtest that are not in xtrain anymore
                X_valid = X_valid.drop(columns=removed_features)
                X_test = X_test.drop(columns=removed_features)

                for estimator in n_estimators_rf:
                    for min_sample_split in min_sample_split_rf:
                        for max_depth in max_depth_rf:
                            rf_clf = RandomForestClassifier(n_estimators=estimator, max_depth=max_depth,
                                                            min_samples_split=min_sample_split)

                            rf_clf.fit(X_train, y_train)

                            y_pred = rf_clf.predict(X_valid)
                            y_pred_train = rf_clf.predict(X_train)

                            if (metrics.f1_score(y_valid,
                                                 y_pred) > f1_scores_rf_valid_year) or f1_scores_rf_valid_year == 0:
                                logloss_rf_valid_year = metrics.log_loss(y_valid, y_pred)
                                fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred)
                                auc_rf_valid_year = metrics.auc(fpr, tpr)
                                f1_scores_rf_valid_year = metrics.f1_score(y_valid, y_pred)
                                MSE_rf_valid_year = metrics.mean_squared_error(y_valid, y_pred)
                                accuracy_rf_valid_year = metrics.accuracy_score(y_valid, y_pred)
                                precision_rf_valid_year, recall_rf_valid_year, fscore, support = metrics.precision_recall_fscore_support(
                                    y_valid, y_pred, average='weighted')
                                rf_probs = rf_clf.predict_proba(X_valid)
                                # keep probabilities for the positive outcome only
                                rf_probs = rf_probs[:, 1]
                                rf_precision, rf_recall, _ = metrics.precision_recall_curve(y_valid, rf_probs)
                                pr_auc_rf_valid_year = metrics.auc(rf_recall, rf_precision)

                                logloss_rf_train_year = metrics.log_loss(y_train, y_pred_train)
                                fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_train)
                                auc_rf_train_year = metrics.auc(fpr, tpr)
                                f1_scores_rf_train_year = metrics.f1_score(y_train, y_pred_train)
                                MSE_rf_train_year = metrics.mean_squared_error(y_train, y_pred_train)
                                accuracy_rf_train_year = metrics.accuracy_score(y_train, y_pred_train)
                                precision_rf_train_year, recall_rf_train_year, fscore, support = metrics.precision_recall_fscore_support(
                                    y_train, y_pred_train, average='weighted')
                                rf_probs = rf_clf.predict_proba(X_train)
                                # keep probabilities for the positive outcome only
                                rf_probs = rf_probs[:, 1]
                                rf_precision, rf_recall, _ = metrics.precision_recall_curve(y_train, rf_probs)
                                pr_auc_rf_train_year = metrics.auc(rf_recall, rf_precision)

                                self.best_rf_param_logloss = {
                                    'n_estimators': estimator,
                                    'max_depth': max_depth,
                                    'min_sample_split': min_sample_split,
                                    'logloss': logloss_rf_valid_year,
                                    'f1_score': f1_scores_rf_valid_year,
                                    'MSE': MSE_rf_valid_year,
                                    'auc': auc_rf_valid_year,
                                    'accuracy': accuracy_rf_valid_year,
                                    'precision': precision_rf_valid_year,
                                    'recall': recall_rf_valid_year,
                                    'pr_auc_score': pr_auc_rf_valid_year,
                                    'logloss_train': logloss_rf_train_year,
                                    'auc_train': auc_rf_train_year,
                                    'f1_score_train': f1_scores_rf_train_year,
                                    'MSE_train': MSE_rf_train_year,
                                    'accuracy_train': accuracy_rf_train_year,
                                    'precision_train': precision_rf_train_year,
                                    'recall_train': recall_rf_train_year,
                                    'pr_auc_score_train': pr_auc_rf_train_year,
                                    'kneighbour': k
                                }

            df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
            df_imputed = KNNAlter(self.df_all, df_imputed, self.best_rf_param_logloss['kneighbour'])
            all_data = df_imputed.to_csv("df_imputed/df_imputed_RF.csv", index=False)

            df_train = df_imputed.loc[df_imputed['year'].isin(train_years)]
            df_valid = df_imputed.loc[df_imputed['year'] == train_years[-1] + 1]
            df_test = df_imputed.loc[df_imputed['year'] == train_years[-1] + 2]
            X_train, y_train = df_train.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_train[
                                   'in_top20_next_year'].copy()
            X_valid, y_valid = df_valid.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_valid[
                                   'in_top20_next_year'].copy()
            X_test, y_test = df_test.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_test[
                                 'in_top20_next_year'].copy()

            # insert previous UPsampling SMOTE code here if needed
            oversample = SMOTE()
            X_train, y_train = oversample.fit_resample(X_train, y_train)

            # feature_selection
            feature_names = list(X_train.columns.values)
            selector = SelectKBest(score_func=mutual_info_classif, k=30)
            X_train = selector.fit_transform(X_train, y_train)  # k optimal?
            mask = selector.get_support()  # list of booleans
            new_features = []
            removed_features = []
            for bool, feature in zip(mask, feature_names):
                if bool:
                    new_features.append(feature)
                else:
                    removed_features.append(feature)
            X_train = pd.DataFrame(X_train, columns=new_features)

            # remove all feautre in xvalid and xtest that are not in xtrain anymore
            X_valid = X_valid.drop(columns=removed_features)
            X_test = X_test.drop(columns=removed_features)

            frames = [X_train, X_valid]
            xtv = pd.concat(frames)
            frames = [y_train, y_valid]
            ytv = pd.concat(frames)
            rf_clf = RandomForestClassifier(n_estimators=self.best_rf_param_logloss['n_estimators'],
                                            max_depth=self.best_rf_param_logloss['max_depth']
                                            , min_samples_split=self.best_rf_param_logloss['min_sample_split'])
            rf_clf.fit(xtv, ytv)

            pickle.dump(rf_clf, open(filename, 'wb'))
            y_pred = rf_clf.predict(X_test)

            # prepare df beforehand and then compare dfall and xtest,ypred and append to df and then write to csv
            # row and y_pred to csv with their names included so gotta compare with df_imputed
            # iterate through dataframe:
            for index, row in X_test.iterrows():
                df_row = df_imputed.loc[
                    (df_imputed[new_features[0]] == row[new_features[0]])
                    & (df_imputed[new_features[1]] == row[
                        new_features[1]])
                    & (df_imputed[new_features[2]] == row[new_features[2]]) & (
                            df_imputed[new_features[3]] == row[new_features[3]])
                    & (df_imputed[new_features[4]] == row[new_features[4]]) & (
                            df_imputed[new_features[5]] == row[new_features[5]])
                    ]

                df_result = df_result.append(df_row, ignore_index=True)
            pred = pd.Series(y_pred, name='Prediction')

            frames = [df_result, pred]
            df_result = pd.concat(frames, axis=1)
            self.df_final_result = self.df_final_result.append(df_result)

            auc_rf_valid += self.best_rf_param_logloss['auc']
            logloss_rf_valid += self.best_rf_param_logloss['logloss']
            f1_scores_rf_valid += self.best_rf_param_logloss['f1_score']
            MSE_rf_valid += self.best_rf_param_logloss['MSE']
            accuracy_rf_valid += self.best_rf_param_logloss['accuracy']
            precision_rf_valid += self.best_rf_param_logloss['precision']
            recall_rf_valid += self.best_rf_param_logloss['recall']
            pr_auc_rf_valid += self.best_rf_param_logloss['pr_auc_score']

            auc_rf_train += self.best_rf_param_logloss['auc_train']
            logloss_rf_train += self.best_rf_param_logloss['logloss_train']
            f1_scores_rf_train += self.best_rf_param_logloss['f1_score_train']
            MSE_rf_train += self.best_rf_param_logloss['MSE_train']
            accuracy_rf_train += self.best_rf_param_logloss['accuracy_train']
            precision_rf_train += self.best_rf_param_logloss['precision_train']
            recall_rf_train += self.best_rf_param_logloss['recall_train']
            pr_auc_rf_train += self.best_rf_param_logloss['pr_auc_score_train']

            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            auc_rf_test_year = metrics.auc(fpr, tpr)
            logloss_rf_test_year = metrics.log_loss(y_test, y_pred)
            f1_scores_rf_test_year = metrics.f1_score(y_test, y_pred)
            MSE_rf_test_year = metrics.mean_squared_error(y_test, y_pred)
            accuracy_rf_test_year = metrics.accuracy_score(y_test, y_pred)
            precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, y_pred,
                                                                                         average='weighted')

            auc_rf_test += auc_rf_test_year
            logloss_rf_test += logloss_rf_test_year
            f1_scores_rf_test += f1_scores_rf_test_year
            MSE_rf_test += MSE_rf_test_year
            accuracy_rf_test += accuracy_rf_test_year
            precision_rf_test = precision + precision_rf_test
            recall_rf_test = recall + recall_rf_test

            rf_probs = rf_clf.predict_proba(X_test)
            # keep probabilities for the positive outcome only
            rf_probs = rf_probs[:, 1]
            # predict class values
            y_pred = rf_clf.predict(X_test)
            rf_precision, rf_recall, _ = metrics.precision_recall_curve(y_test, rf_probs)
            pr_auc_rf_test_year = metrics.auc(rf_recall, rf_precision)
            pr_auc_rf_test += pr_auc_rf_test_year

            df_row = {"yearmodel": train_years[-1] + 3, "bestk": self.best_rf_param_logloss["kneighbour"]}
            df_bestk = df_bestk.append(df_row, ignore_index=True)

            df_row = {"model": "Random Forest Training set", "year": train_years[-1] + 3,
                      "f1_score": self.best_rf_param_logloss['f1_score_train'],
                      "precision": self.best_rf_param_logloss['precision_train'],
                      "recall": self.best_rf_param_logloss['recall_train'],
                      "logloss": self.best_rf_param_logloss['logloss_train'],
                      "roc auc score": self.best_rf_param_logloss['auc_train'],
                      "MSE": self.best_rf_param_logloss['MSE_train'],
                      "accuracy": self.best_rf_param_logloss['accuracy_train'],
                      "pr auc score": self.best_rf_param_logloss['pr_auc_score_train']}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            df_row = {"model": "Random Forest Validation set", "year": train_years[-1] + 3,
                      "f1_score": self.best_rf_param_logloss['f1_score'],
                      "precision": self.best_rf_param_logloss['precision'],
                      "recall": self.best_rf_param_logloss['recall'],
                      "logloss": self.best_rf_param_logloss['logloss'],
                      "roc auc score": self.best_rf_param_logloss['auc'],
                      "MSE": self.best_rf_param_logloss['MSE'],
                      "accuracy": self.best_rf_param_logloss['accuracy'],
                      "pr auc score": self.best_rf_param_logloss['pr_auc_score']}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            df_row = {"model": "Random Forest Test set", "year": train_years[-1] + 3,
                      "f1_score": f1_scores_rf_test_year,
                      "precision": precision, "recall": recall,
                      "logloss": logloss_rf_test_year, "roc auc score": auc_rf_test_year,
                      "MSE": MSE_rf_test_year,
                      "accuracy": accuracy_rf_test_year,
                      "pr auc score": pr_auc_rf_test_year}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            self.df_metric_yearly_results.to_csv("metric_yearly_results/df_metric_yearly_results_rf.csv", index=False)

            if train_years[-1] == 2019: #here
                break
            else:
                train_years.append(train_years[-1] + 1)
                count += 1

        all_data = df_bestk.to_csv("df_bestk/df_bestk_RF.csv", index=False)

        all_data = self.df_final_result.to_csv("df_results/df_results_RF.csv", index=False)

        auc_rf_train = auc_rf_train / count
        logloss_rf_train = logloss_rf_train / count
        f1_scores_rf_train = f1_scores_rf_train / count
        MSE_rf_train = MSE_rf_train / count
        accuracy_rf_train = accuracy_rf_train / count
        precision_rf_train = precision_rf_train / count
        recall_rf_train = recall_rf_train / count
        pr_auc_rf_train = pr_auc_rf_train / count

        auc_rf_valid = auc_rf_valid / count
        logloss_rf_valid = logloss_rf_valid / count
        f1_scores_rf_valid = f1_scores_rf_valid / count
        MSE_rf_valid = MSE_rf_valid / count
        accuracy_rf_valid = accuracy_rf_valid / count
        precision_rf_valid = precision_rf_valid / count
        recall_rf_valid = recall_rf_valid / count
        pr_auc_rf_valid = pr_auc_rf_valid / count

        auc_rf_test = auc_rf_test / count
        logloss_rf_test = logloss_rf_test / count
        f1_scores_rf_test = f1_scores_rf_test / count
        MSE_rf_test = MSE_rf_test / count
        accuracy_rf_test = accuracy_rf_test / count
        precision_rf_test = precision_rf_test / count
        recall_rf_test = recall_rf_test / count
        pr_auc_rf_test = pr_auc_rf_test / count

        # plot the precision-recall curves
        pyplot.plot(rf_recall, rf_precision, marker='.', label='RF')

        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()

        end = time.time()
        print(end - start)
        print('done')

        df_row = {"model": "Random Forest Training set",
                  "f1_score": f1_scores_rf_train,
                  "precision": precision_rf_train, "recall": recall_rf_train,
                  "logloss": logloss_rf_train, "roc auc score": auc_rf_train,
                  "MSE": MSE_rf_train, "accuracy": accuracy_rf_train,
                  "pr auc score": pr_auc_rf_train}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        df_row = {"model": "Random Forest Validation set",
                  "f1_score": f1_scores_rf_valid,
                  "precision": precision_rf_valid, "recall": recall_rf_valid,
                  "logloss": logloss_rf_valid, "roc auc score": auc_rf_valid,
                  "MSE": MSE_rf_valid, "accuracy": accuracy_rf_valid,
                  "pr auc score": pr_auc_rf_valid}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        df_row = {"model": "Random Forest Test set",
                  "f1_score": f1_scores_rf_test,
                  "precision": precision_rf_test, "recall": recall_rf_test,
                  "logloss": logloss_rf_test, "roc auc score": auc_rf_test,
                  "MSE": MSE_rf_test, "accuracy": accuracy_rf_test,
                  "pr auc score": pr_auc_rf_test}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        self.df_metric_results.to_csv("metric_results/df_metric_results_rf.csv", index=False)

class LogReg(Model):
    def __init__(self):
        Model.__init__(self)
        self.best_lr_param_logloss = {
            'weight': 0,
            'logloss': 0,
            'auc': 0,
            'f1_score': 0,
            'MSE': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'pr_auc_score': 0,
            'logloss_train': 0,
            'auc_train': 0,
            'f1_score_train': 0,
            'MSE_train': 0,
            'accuracy_train': 0,
            'precision_train': 0,
            'recall_train': 0,
            'pr_auc_score_train': 0,
            'kneighbour': 0
        }

    def run(self):
        start = time.time()
        
        # weight_lr = ["balanced",{0:100,1:1}, {0:10,1:1}, {0:1,1:1}, {0:1,1:10}, {0:1,1:100}]
        weight_lr = ["balanced"]

        auc_lr_train, logloss_lr_train, f1_scores_lr_train, MSE_lr_train, accuracy_lr_train, precision_lr_train, recall_lr_train, pr_auc_lr_train = 0, 0, 0, 0, 0, 0, 0, 0
        auc_lr_valid, logloss_lr_valid, f1_scores_lr_valid, MSE_lr_valid, accuracy_lr_valid, precision_lr_valid, recall_lr_valid, pr_auc_lr_valid = 0, 0, 0, 0, 0, 0, 0, 0
        auc_lr_test, logloss_lr_test, f1_scores_lr_test, MSE_lr_test, accuracy_lr_test, precision_lr_test, recall_lr_test, pr_auc_lr_test = 0, 0, 0, 0, 0, 0, 0, 0

        filename = 'saved_models/lr_model.sav'

        train_years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
        # train_years = [2009]
        count = 1
        # lr_recall_all = np.array([])
        # lr_precision_all = np.array([])

        #numofneighbours = [1, 3, 5, 7, 9, 11]
        numofneighbours = [1]

        df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
        df_bestk = pd.DataFrame([], columns=["yearmodel", "bestk"])

        for year in train_years:
            auc_lr_valid_year, logloss_lr_valid_year, f1_scores_lr_valid_year, MSE_lr_valid_year, accuracy_lr_valid_year, \
            precision_lr_valid_year, recall_lr_valid_year, pr_auc_lr_valid_year = 0, 0, 0, 0, 0, 0, 0, 0

            auc_lr_train_year, logloss_lr_train_year, f1_scores_lr_train_year, MSE_lr_train_year, accuracy_lr_train_year, \
            precision_lr_train_year, recall_lr_train_year, pr_auc_lr_train_year = 0, 0, 0, 0, 0, 0, 0, 0
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
                                              "age", "career year","rider_bmi", "win_ratio_slope",
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
                                              "in_top20_next_year"])

            for k in numofneighbours:
                df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
                df_imputed = KNNAlter(self.df_all, df_imputed, k)
                # all_data = df_imputed.to_csv("df_imputed.csv", index=False)

                df_train = df_imputed.loc[df_imputed['year'].isin(train_years)]
                df_valid = df_imputed.loc[df_imputed['year'] == train_years[-1] + 1]
                df_test = df_imputed.loc[df_imputed['year'] == train_years[-1] + 2]
                X_train, y_train = df_train.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_train[
                                       'in_top20_next_year'].copy()
                X_valid, y_valid = df_valid.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_valid[
                                       'in_top20_next_year'].copy()
                X_test, y_test = df_test.drop(
                    columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_test[
                                     'in_top20_next_year'].copy()

                # insert previous UPsampling SMOTE code here if needed
                oversample = SMOTE()
                X_train, y_train = oversample.fit_resample(X_train, y_train)

                # feature_selection
                feature_names = list(X_train.columns.values)
                selector = SelectKBest(score_func=mutual_info_classif, k=30)
                X_train = selector.fit_transform(X_train, y_train)  # k optimal?
                mask = selector.get_support()  # list of booleans
                new_features = []
                removed_features = []
                for bool, feature in zip(mask, feature_names):
                    if bool:
                        new_features.append(feature)
                    else:
                        removed_features.append(feature)
                X_train = pd.DataFrame(X_train, columns=new_features)

                # remove all feautre in xvalid and xtest that are not in xtrain anymore
                X_valid = X_valid.drop(columns=removed_features)
                X_test = X_test.drop(columns=removed_features)

                for weight in weight_lr:

                    lr_clf = LogisticRegression(class_weight=weight, max_iter=5000, solver='saga')
                    lr_clf.fit(X_train, y_train)
                    y_pred = lr_clf.predict(X_valid)
                    y_pred_train = lr_clf.predict(X_train)

                    if (metrics.f1_score(y_valid, y_pred) > f1_scores_lr_valid_year) or f1_scores_lr_valid_year == 0:
                        logloss_lr_valid_year = metrics.log_loss(y_valid, y_pred)
                        fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred)
                        auc_lr_valid_year = metrics.auc(fpr, tpr)
                        f1_scores_lr_valid_year = metrics.f1_score(y_valid, y_pred)
                        MSE_lr_valid_year = metrics.mean_squared_error(y_valid, y_pred)
                        accuracy_lr_valid_year = metrics.accuracy_score(y_valid, y_pred)
                        precision_lr_valid_year, recall_lr_valid_year, fscore, support = metrics.precision_recall_fscore_support(
                            y_valid, y_pred, average='weighted')
                        lr_probs = lr_clf.predict_proba(X_valid)
                        # keep probabilities for the positive outcome only
                        lr_probs = lr_probs[:, 1]
                        lr_precision, lr_recall, _ = metrics.precision_recall_curve(y_valid, lr_probs)
                        pr_auc_lr_valid_year = metrics.auc(lr_recall, lr_precision)

                        logloss_lr_train_year = metrics.log_loss(y_train, y_pred_train)
                        fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_train)
                        auc_lr_train_year = metrics.auc(fpr, tpr)
                        f1_scores_lr_train_year = metrics.f1_score(y_train, y_pred_train)
                        MSE_lr_train_year = metrics.mean_squared_error(y_train, y_pred_train)
                        accuracy_lr_train_year = metrics.accuracy_score(y_train, y_pred_train)
                        precision_lr_train_year, recall_lr_train_year, fscore, support = metrics.precision_recall_fscore_support(
                            y_train, y_pred_train, average='weighted')
                        lr_probs = lr_clf.predict_proba(X_train)
                        # keep probabilities for the positive outcome only
                        lr_probs = lr_probs[:, 1]
                        lr_precision, lr_recall, _ = metrics.precision_recall_curve(y_train, lr_probs)
                        pr_auc_lr_train_year = metrics.auc(lr_recall, lr_precision)

                        self.best_lr_param_logloss = {
                            'weight': weight,
                            'logloss': logloss_lr_valid_year,
                            'f1_score': f1_scores_lr_valid_year,
                            'MSE': MSE_lr_valid_year,
                            'auc': auc_lr_valid_year,
                            'accuracy': accuracy_lr_valid_year,
                            'precision': precision_lr_valid_year,
                            'recall': recall_lr_valid_year,
                            'pr_auc_score': pr_auc_lr_valid_year,
                            'logloss_train': logloss_lr_train_year,
                            'auc_train': auc_lr_train_year,
                            'f1_score_train': f1_scores_lr_train_year,
                            'MSE_train': MSE_lr_train_year,
                            'accuracy_train': accuracy_lr_train_year,
                            'precision_train': precision_lr_train_year,
                            'recall_train': recall_lr_train_year,
                            'pr_auc_score_train': pr_auc_lr_train_year,
                            'kneighbour': k
                        }

            df_imputed = self.df_all.drop(columns=['rider_name', 'year'])
            df_imputed = KNNAlter(self.df_all, df_imputed, self.best_lr_param_logloss['kneighbour'])
            all_data = df_imputed.to_csv("df_imputed/df_imputed_LR.csv", index=False)

            df_train = df_imputed.loc[df_imputed['year'].isin(train_years)]
            df_valid = df_imputed.loc[df_imputed['year'] == train_years[-1] + 1]
            df_test = df_imputed.loc[df_imputed['year'] == train_years[-1] + 2]
            X_train, y_train = df_train.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_train[
                                   'in_top20_next_year'].copy()
            X_valid, y_valid = df_valid.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_valid[
                                   'in_top20_next_year'].copy()
            X_test, y_test = df_test.drop(
                columns=['in_top20_next_year', 'year', 'rider_name']).copy(), df_test[
                                 'in_top20_next_year'].copy()

            # insert previous UPsampling SMOTE code here if needed
            oversample = SMOTE()
            X_train, y_train = oversample.fit_resample(X_train, y_train)

            # feature_selection
            feature_names = list(X_train.columns.values)
            selector = SelectKBest(score_func=mutual_info_classif, k=30)
            X_train = selector.fit_transform(X_train, y_train)  # k optimal?
            mask = selector.get_support()  # list of booleans
            new_features = []
            removed_features = []
            for bool, feature in zip(mask, feature_names):
                if bool:
                    new_features.append(feature)
                else:
                    removed_features.append(feature)
            X_train = pd.DataFrame(X_train, columns=new_features)

            # remove all feautre in xvalid and xtest that are not in xtrain anymore
            X_valid = X_valid.drop(columns=removed_features)
            X_test = X_test.drop(columns=removed_features)

            frames = [X_train, X_valid]
            xtv = pd.concat(frames)
            frames = [y_train, y_valid]
            ytv = pd.concat(frames)
            lr_clf = LogisticRegression(class_weight=self.best_lr_param_logloss['weight'], solver='saga')
            lr_clf.fit(xtv, ytv)

            pickle.dump(lr_clf, open(filename, 'wb'))
            y_pred = lr_clf.predict(X_test)

            # prepare df beforehand and then compare dfall and xtest,ypred and append to df and then write to csv
            # row and y_pred to csv with their names included so gotta compare with df_imputed
            # iterate through dataframe:
            for index, row in X_test.iterrows():
                df_row = df_imputed.loc[
                    (df_imputed[new_features[0]] == row[new_features[0]])
                    & (df_imputed[new_features[1]] == row[
                        new_features[1]])
                    & (df_imputed[new_features[2]] == row[new_features[2]]) & (
                            df_imputed[new_features[3]] == row[new_features[3]])
                    & (df_imputed[new_features[4]] == row[new_features[4]]) & (
                            df_imputed[new_features[5]] == row[new_features[5]])
                    ]

                df_result = df_result.append(df_row, ignore_index=True)
            pred = pd.Series(y_pred, name='Prediction')

            frames = [df_result, pred]
            df_result = pd.concat(frames, axis=1)
            self.df_final_result = self.df_final_result.append(df_result)

            auc_lr_valid += self.best_lr_param_logloss['auc']
            logloss_lr_valid += self.best_lr_param_logloss['logloss']
            f1_scores_lr_valid += self.best_lr_param_logloss['f1_score']
            MSE_lr_valid += self.best_lr_param_logloss['MSE']
            accuracy_lr_valid += self.best_lr_param_logloss['accuracy']
            precision_lr_valid += self.best_lr_param_logloss['precision']
            recall_lr_valid += self.best_lr_param_logloss['recall']
            pr_auc_lr_valid += self.best_lr_param_logloss['pr_auc_score']

            auc_lr_train += self.best_lr_param_logloss['auc_train']
            logloss_lr_train += self.best_lr_param_logloss['logloss_train']
            f1_scores_lr_train += self.best_lr_param_logloss['f1_score_train']
            MSE_lr_train += self.best_lr_param_logloss['MSE_train']
            accuracy_lr_train += self.best_lr_param_logloss['accuracy_train']
            precision_lr_train += self.best_lr_param_logloss['precision_train']
            recall_lr_train += self.best_lr_param_logloss['recall_train']
            pr_auc_lr_train += self.best_lr_param_logloss['pr_auc_score_train']

            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            auc_lr_test_year = metrics.auc(fpr, tpr)
            logloss_lr_test_year = metrics.log_loss(y_test, y_pred)
            f1_scores_lr_test_year = metrics.f1_score(y_test, y_pred)
            MSE_lr_test_year = metrics.mean_squared_error(y_test, y_pred)
            accuracy_lr_test_year = metrics.accuracy_score(y_test, y_pred)
            precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, y_pred,
                                                                                         average='weighted')

            auc_lr_test += auc_lr_test_year
            logloss_lr_test += logloss_lr_test_year
            f1_scores_lr_test += f1_scores_lr_test_year
            MSE_lr_test += MSE_lr_test_year
            accuracy_lr_test += accuracy_lr_test_year
            precision_lr_test = precision + precision_lr_test
            recall_lr_test = recall + recall_lr_test

            lr_probs = lr_clf.predict_proba(X_test)
            # keep probabilities for the positive outcome only
            lr_probs = lr_probs[:, 1]
            # predict class values
            y_pred = lr_clf.predict(X_test)
            lr_precision, lr_recall, _ = metrics.precision_recall_curve(y_test, lr_probs)
            pr_auc_lr_test_year = metrics.auc(lr_recall, lr_precision)
            pr_auc_lr_test += pr_auc_lr_test_year

            df_row = {"yearmodel": train_years[-1] + 3, "bestk": self.best_lr_param_logloss["kneighbour"]}
            df_bestk = df_bestk.append(df_row, ignore_index=True)

            df_row = {"model": "Logistic Regression Training set", "year": train_years[-1] + 3,
                      "f1_score": self.best_lr_param_logloss['f1_score_train'],
                      "precision": self.best_lr_param_logloss['precision_train'],
                      "recall": self.best_lr_param_logloss['recall_train'],
                      "logloss": self.best_lr_param_logloss['logloss_train'],
                      "roc auc score": self.best_lr_param_logloss['auc_train'],
                      "MSE": self.best_lr_param_logloss['MSE_train'],
                      "accuracy": self.best_lr_param_logloss['accuracy_train'],
                      "pr auc score": self.best_lr_param_logloss['pr_auc_score_train']}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            df_row = {"model": "Logistic Regression Validation set", "year": train_years[-1] + 3,
                      "f1_score": self.best_lr_param_logloss['f1_score'],
                      "precision": self.best_lr_param_logloss['precision'],
                      "recall": self.best_lr_param_logloss['recall'],
                      "logloss": self.best_lr_param_logloss['logloss'],
                      "roc auc score": self.best_lr_param_logloss['auc'],
                      "MSE": self.best_lr_param_logloss['MSE'],
                      "accuracy": self.best_lr_param_logloss['accuracy'],
                      "pr auc score": self.best_lr_param_logloss['pr_auc_score']}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            df_row = {"model": "Logistic Regression Test set", "year": train_years[-1] + 3,
                      "f1_score": f1_scores_lr_test_year,
                      "precision": precision, "recall": recall,
                      "logloss": logloss_lr_test_year, "roc auc score": auc_lr_test_year,
                      "MSE": MSE_lr_test_year,
                      "accuracy": accuracy_lr_test_year,
                      "pr auc score": pr_auc_lr_test_year}
            self.df_metric_yearly_results = self.df_metric_yearly_results.append(df_row, ignore_index=True)
            self.df_metric_yearly_results.to_csv("metric_yearly_results/df_metric_yearly_results_lr.csv", index=False)

            if train_years[-1] == 2019: #here
                break
            else:
                train_years.append(train_years[-1] + 1)
                count += 1

        all_data = df_bestk.to_csv("df_bestk/df_bestk_LR.csv", index=False)

        all_data = self.df_final_result.to_csv("df_results/df_results_LR.csv", index=False)

        auc_lr_train = auc_lr_train / count
        logloss_lr_train = logloss_lr_train / count
        f1_scores_lr_train = f1_scores_lr_train / count
        MSE_lr_train = MSE_lr_train / count
        accuracy_lr_train = accuracy_lr_train / count
        precision_lr_train = precision_lr_train / count
        recall_lr_train = recall_lr_train / count
        pr_auc_lr_train = pr_auc_lr_train / count

        auc_lr_valid = auc_lr_valid / count
        logloss_lr_valid = logloss_lr_valid / count
        f1_scores_lr_valid = f1_scores_lr_valid / count
        MSE_lr_valid = MSE_lr_valid / count
        accuracy_lr_valid = accuracy_lr_valid / count
        precision_lr_valid = precision_lr_valid / count
        recall_lr_valid = recall_lr_valid / count
        pr_auc_lr_valid = pr_auc_lr_valid / count

        auc_lr_test = auc_lr_test / count
        logloss_lr_test = logloss_lr_test / count
        f1_scores_lr_test = f1_scores_lr_test / count
        MSE_lr_test = MSE_lr_test / count
        accuracy_lr_test = accuracy_lr_test / count
        precision_lr_test = precision_lr_test / count
        recall_lr_test = recall_lr_test / count
        pr_auc_lr_test = pr_auc_lr_test / count

        # plot the precision-recall curves
        pyplot.plot(lr_recall, lr_precision, marker='.', label='LogReg')

        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()

        end = time.time()
        print(end - start)
        print('done')

        df_row = {"model": "Logistic Regression Training set",
                  "f1_score": f1_scores_lr_train,
                  "precision": precision_lr_train, "recall": recall_lr_train,
                  "logloss": logloss_lr_train, "roc auc score": auc_lr_train,
                  "MSE": MSE_lr_train,
                  "accuracy": accuracy_lr_train,
                  "pr auc score": pr_auc_lr_train}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        df_row = {"model": "Logistic Regression Validation set",
                  "f1_score": f1_scores_lr_valid,
                  "precision": precision_lr_valid, "recall": recall_lr_valid,
                  "logloss": logloss_lr_valid, "roc auc score": auc_lr_valid,
                  "MSE": MSE_lr_valid, "accuracy": accuracy_lr_valid,
                  "pr auc score": pr_auc_lr_valid}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        df_row = {"model": "Logistic Regression Test set",
                  "f1_score": f1_scores_lr_test,
                  "precision": precision_lr_test, "recall": recall_lr_test,
                  "logloss": logloss_lr_test, "roc auc score": auc_lr_test,
                  "MSE": MSE_lr_test, "accuracy": accuracy_lr_test,
                  "pr auc score": pr_auc_lr_test}
        self.df_metric_results = self.df_metric_results.append(df_row, ignore_index=True)
        self.df_metric_results.to_csv("metric_results/df_metric_results_lr.csv", index=False)
