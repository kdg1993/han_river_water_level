# han_river_water_level
Han River Water Level Prediction AI Competition for  Flood Safety of Paldang Dam hosted by Korea Hydro &amp; Nuclear Power Co., Ltd. at DACON

---
## Experiment log
* Submission date : 2022-08-11 14:43:25
* Model : Sklearn's RandomForestRegressor
* Dataset
  * Dimension : 2D(tabular) | 44 columns
  * Consider t ~ t-21(unit time)
  * Selected features : 'fw_1018683', 'fw_1019630' because of its less missing point in test time range
  * Temporal range : 2012, 2013, 2016, 2017, 2018, 2020, 2022 (Selected by features' variance)
* Target
  * Four classes of y for t+1
* Score
  * Validation score : around 4.27 (RMSE/R^2)
  * Public score : 4.219 (RMSE/R^2)
* Experiment review
  * The score is not low enough. However, it is hard to widen the past time range for giving the model more data


* Submission date : 2022-08-13 23:07:29
* Model : LightGBM Regressor
* Dataset
  * Dimension : 2D(tabular) | 50 columns
  * Consider t ~ t-48(unit time)
  * Selected features : 'fw_1018683', 'fw_1019630' because of its less missing point in test time range
  * Temporal range : 2012 ~ 2022 (Selected all)
* Target
  * Four classes of y for t+1
* Score
  * Validation score : around 4.75 (RMSE/R^2)
  * Public score : 5.425 (RMSE/R^2)
* Experiment review
  * LightGBM was worse than random forest in both validation and public score but speed was much faster
  * Due to the fast speed, hyperparameter tuning was done enough. Thus, more data seems to be needed to overcome the low scores.

* Submission date : 2022-08-17 16:16:56	
* Model : XGBoostRandomForestRegressor
* Dataset
  * Dimension : 2D(tabular) | 66 columns
  * Consider t ~ t-21(unit time) | best among 9, 12, 15, 18, 21
  * Selected features
    * 'fw_1018683', 'fw_1019630' because of its less missing point in test time range
    * Consider 'fw_1018662' because xgboost algorithm can handle missing value automatically 
  * Temporal range : 2012, 2013, 2016, 2017, 2018, 2020, 2022 (Selected by features' variance)
  * Note
    * Did not remove instances that have nan in X (due to the algorithm)
    * Remove instance when there is nan in y
* Target
  * Four classes of y for t+1
* Score
  * Validation score : around 3.3177 (RMSE/R^2)
  * Public score : 3.5764 (RMSE/R^2)
* Experiment review
  * Unlike sklearn's random forest, public score was higher than validation score
    * Suspected reason : Missing ratio of column fw_1018662 is higher in X_test than X_train and X_val 
  * More data always works even with the missing values
  * XGBoost algorithm can handle the missing value quite well
