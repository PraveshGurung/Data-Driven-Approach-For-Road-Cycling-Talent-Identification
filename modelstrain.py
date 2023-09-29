from models3 import XGBoost,LightGBM,CatBoost,RandomForest,LogReg
from ngboost_performance_curve import NGBModel
import logging
if __name__ == "__main__":
    xgb = XGBoost()
    xgb.run()
    cgb = CatBoost()
    cgb.run()
    lgb = LightGBM()
    lgb.run()
    rf = RandomForest()
    rf.run()
    lr = LogReg()
    lr.run()
    #ngb = NGBModel()
    #ngb.run()
    #ngb.plot_performance_curve(2017,10)
    #ngb.plot_performance_curve_specific_rider("tom-boonen")