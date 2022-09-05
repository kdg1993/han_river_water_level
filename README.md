# han_river_water_level
Han River Water Level Prediction AI Competition for  Flood Safety of Paldang Dam hosted by Korea Hydro &amp; Nuclear Power Co., Ltd. at DACON

---
## Experiment log
### Submission date : 2022-08-11 14:43:25
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
* Strategy
  * Random forest could be robust for the noisy data
  * Features started with 'fw' show a similar temporal pattern to the target
  * Among the four 'fw' featuers, only two features show a small number of nan in both train and test period
* Experiment review
  * The score is not low enough. However, it is hard to widen the past time range for giving the model more data
---
### Submission date : 2022-08-13 23:07:29
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
* Strategy
  * Boosting algorithms usually show high accuracy when good quality data is given
  * LightGBM is much quicker than RandomForest. Thus, hyperparameter tuning has a lot more range and chance
  * Since LightGBM is fast, we can use a bigger dataset (wider range of past time windows)
* Experiment review
  * LightGBM was worse than random forest in both validation and public score but speed was much faster
  * Due to the fast speed, hyperparameter tuning was done enough. Thus, more data seems to be needed to overcome the low scores.
---
### Submission date : 2022-08-17 16:16:56	
* Model : XGBoostRandomForestRegressor
* Dataset
  * Dimension : 2D(tabular) | 66 columns
  * Consider t ~ t-21(unit time) | best among 9, 12, 15, 18, 21
  * Selected features (3)
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
* Strategy
  * RandomForest was better than LightGBM
  * XGBoost algorithms can handle missing values without any preprocessing
    * Thus, it can make it feature fw_1018662 that has many missing points in test set
* Experiment review
  * Unlike sklearn's random forest, public score was higher than validation score
    * Suspected reason : Missing ratio of column fw_1018662 is higher in X_test than X_train and X_val 
  * More data always works even with the missing values
  * XGBoost algorithm can handle the missing value quite well
---
### Submission date : 2022-08-18 23:16:04
* Model : XGBoostRandomForestRegressor
* Dataset
  * Dimension : 2D(tabular) | 117 columns
  * Consider t ~ t-12(unit time) | best among 6, 12, 18
  * Selected features (9)
    * All features in water level data except one fw feature that has a lot of missing
    * Consider 'fw_1018662' because xgboost algorithm can handle missing value automatically 
  * Temporal range : 2012, 2013, 2016, 2017, 2018, 2020, 2022 (Selected by features' variance)
  * Note
    * Did not remove instances that have nan in X (due to the algorithm)
    * Remove instance when there is nan in y
* Target
  * Four classes of y for t+1
* Score
  * Validation score : around 2.1 (RMSE/R^2)
  * Public score : 4.0094 (RMSE/R^2)
* Strategy
  * Features in water level data which are not fw_ have a small number of missing values. Although their temporal patterns are not similar to the targets but could be informative for the model
  * RandomForest feature sampling might effective for a large number of features
* Experiment review
  * Validation and Public scores indicate overfitting
  * Careless feature addition was counterproductive
---
### Submission date : 2022-08-21 19:42:42
* Model : XGBoostRandomForestRegressor
* Dataset
  * Dimension : 2D(tabular) | 65 columns
  * Consider t ~ t-12(unit time) | best among 6, 12
  * Selected features (5)
    * All features in water level data except one fw feature that has a lot of missing
    * Consider 'fw_1018662' because xgboost algorithm can handle missing value automatically
    * Based on Spearman's correlation, features 'inf', 'tototf' showed relatively higher values with target. Thus, we expected these features have enough information to guess target 
  * Temporal range : 2012, 2013, 2016, 2017, 2018, 2020, 2022 (Selected by features' variance)
  * Note
    * Did not remove instances that have nan in X (due to the algorithm)
    * Remove instance when there is nan in y
* Target
  * Four classes of y for t+1
* Score
  * Validation score : around 2.850 (RMSE/R^2)
  * Public score : 3.8271 (RMSE/R^2)
* Strategy
  * Based on Spearman's correlation, features 'inf', 'tototf' showed relatively higher values with target. Thus, we expected these features have enough information to guess target
* Experiment review
  * Overfitting is suspected
  * Hard to infer the reason of overfit. Monotonic relation might be too simple to overcome overfit?
---
### Submission date : 2022-08-22 00:31:09
* Model : XGBoostRandomForestRegressor
* Dataset
  * Dimension : 2D(tabular) | 49 columns
  * Consider t ~ t-6(unit time) | best among 6
  * Selected features (7)
    * All features in water level data except one fw feature that has a lot of missing
    * Consider 'fw_1018662' because xgboost algorithm can handle missing value automatically
    * Based on three fw_ features, add the other features from the water level dataset except 'inf', 'tototf' that might be related to overfitting
  * Temporal range : 2012, 2013, 2016, 2017, 2018, 2020, 2022 (Selected by features' variance)
  * Note
    * Did not remove instances that have nan in X (due to the algorithm)
    * Remove instance when there is nan in y
* Target
  * Four classes of y for t+1
* Score
  * Validation score : around 2.6 (RMSE/R^2)
  * Public score : 4.1245 (RMSE/R^2)
* Strategy
  * Features 'inf', 'tototf' which were selected based on the decision with Spearman's correlation caused overfitting. If it indicates simple monotonic relation causes overfit then features that show less monotonic relation might be useful to add 
* Experiment review
  * More overfitting then add 'inf', 'tototf'
  * Except for the 'fw' features, features from the water level dataset caused overfitting
  * If simple feature selection is not working, changing a model or handling missing values better might be a way
---
### Submission date : 2022-08-22 23:51:26
* Model : ExtraTreeRegressor
* Dataset
  * Dimension : 2D(tabular) | 50 columns
  * Consider t ~ t-12(unit time) | best among 6, 12
  * Selected features (2)
    * Use only two 'fw' features that has few missing in train and no missing in test
  * Temporal range : 2012, 2013, 2016, 2017, 2018, 2020, 2022 (Selected by features' variance)
  * Note
    * No implementation for handling missing value
    * Remove instance when there is nan in y
* Target
  * Four classes of y for t+1
* Score
  * Validation score : around 4.65 (RMSE/R^2)
  * Public score : 4.4191 (RMSE/R^2)
* Strategy
  * ExtraTreeRegressor assign more randomness than Vanilla RandomForestRegressor and it might be better than vanilla RF
* Experiment review
  * Not show a sign of overfitting
  * Slightly worse than Vanilla RF in Sklearn for both validation and public. However, the different scores might be a clue to the significantly different model structures of ETR, which makes it worth trying more
---
### Submission date : 2022-08-25 23:35:10
* Model : ExtraTreeRegressor
* Dataset
  * Dimension : 2D(tabular) | 63 columns
  * Consider t ~ t-6(unit time) | best among 6
  * Selected features (9)
    * All features in water level data except one fw feature that has a lot of missing
  * Temporal range : 2012 ~ 2022 (Selected all)
  * Note
    * Fill missing every missing value as -9999
    * Remove instance when there is nan in y
* Target
  * Four classes of y for t+1
* Score
  * Validation score : around 2.18 (RMSE/R^2)
  * Public score : 3.7479 (RMSE/R^2)
* Strategy
  * ETR's randomness might be a key to overcoming the overfitting that RF showed with all features
  * Fill missing by an extreme valued integer is simple but has no contradict 
* Experiment review
  * Overfitting is suspected
  * The gap between validation and public score is slightly better than the case of XGBRandomForest with all features, but hard to deny overfitting
---
### Submission date : 2022-08-26 21:33:48
* Model : LightGBM Regressor
* Dataset
  * Dimension : 2D(tabular) | 117 columns
  * Consider t ~ t-12(unit time) | best among 6, 12
  * Selected features (9)
    * All features in water level data except one fw feature that has a lot of missing
  * Temporal range : 2012 ~ 2022 (Selected all)
  * Note
    * Fill missing every missing value as -9999
    * Remove instance when there is nan in y
* Target
  * Four classes of y for t+1
* Score
  * Validation score : around 2.99 (RMSE/R^2)
  * Public score : 5.2867 (RMSE/R^2)
* Strategy
  * LightGBM's fast speed can make it possible to search hyperparameters for more times than RF
  * Fill missing values with simple int(-9999) making more features usable with LightGBM 
* Experiment review
  * Overfitting is suspected
  * LightGBM is known to tend to overfit easily if data is small, which is a suspected reason of overfit in this case