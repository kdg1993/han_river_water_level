# han_river_water_level
Han River Water Level Prediction AI Competition for  Flood Safety of Paldang Dam hosted by Korea Hydro &amp; Nuclear Power Co., Ltd. at DACON

---
## Chronological Experiments Summary
![submission_scores_lineplot](https://user-images.githubusercontent.com/50651319/194519092-2748cb82-4a99-4976-bc6c-503d868dda4e.png)
1. RandomForest for noisy data and select a few fine features &rightarrow; A good start but RF is too slow
2. LightGBM for much faster experiments with bigger data &rightarrow; Worse score. More data is needed
3. Add a fine feature by auto nan handling XGBoost &rightarrow; It worked. More data need to be tested
4. Put many more features &rightarrow; Much better validation score but overfitting. No careless features addition
5. Feature selction by Spearman's $\rho$ &rightarrow; Overfitting. Monotonic relation is too simple to overcom overfitting?
6. Select features with less monotonic by Spearman's $\rho$ &rightarrow; More overfitting. Find ways other than feature selection
7. ExtraTreeRegressor's high randomness with a few good features &rightarrow; Low score but not overfitted. It's worth more trying
8. ETR + features overfitted by RF &rightarrow; Overfitting. Structural difference might have not been enough
9. Fill missing by simple integer with LightGBM &rightarrow; Overfitting. Data size might not enough to avoid overfitting with LightGBM
10. A few good features + RF + simple int imputation &rightarrow; (Final best) There is little difference from submission #3
11. Apply simple 4-fold validation for validation score reliability &rightarrow; Overfitting. Feature selection seems to be needed
12. Change model from ETR to vanilla RF &rightarrow; Not a significant difference. Model structure is not a key
13. Reduce the number of features &rightarrow; Overfitting lessened but no score improvement

---
## Experiment Log
### Submission 1 date : 2022-08-11 14:43:25
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
  * The score is not good enough. However, it is hard to widen the past time range for giving the model more data due to the running time and memory issue
---
### Submission 2 date : 2022-08-13 23:07:29
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
### Submission 3 date : 2022-08-17 16:16:56	
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
### Submission 4 date : 2022-08-18 23:16:04
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
### Submission 5 date : 2022-08-21 19:42:42
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
### Submission 6 date : 2022-08-22 00:31:09
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
  * If simple feature selection is not working, changing a model or better way to handle missing value might be a way
---
### Submission 7 date : 2022-08-22 23:51:26
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
### Submission 8 date : 2022-08-25 23:35:10
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
### Submission 9 date : 2022-08-26 21:33:48
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
---
### Submission 10 date : 2022-08-27 19:19:35 [Final best]
* Model : XGBoostRandomForestRegressor
* Dataset
  * Dimension : 2D(tabular) | 57 columns
  * Consider t ~ t-18(unit time) | best among 1, 6, 12, 18
  * Selected features (3)
    * 'fw_1018683', 'fw_1019630' because of its less missing point in test time range
    * Consider 'fw_1018662' because xgboost algorithm can handle missing value automatically 
  * Temporal range : 2012 ~ 2022 (Selected all)
  * Note
    * Fill missing every missing value as -9999
    * Remove instance when there is nan in y
* Target
  * Four classes of y for t+1
* Score
  * Validation score : around 3.08 (RMSE/R^2)
  * Public score : 3.5233 (RMSE/R^2)
* Strategy
  * Fill missing values with simple int(-9999). It implicitly orders the model not to use the automatical missing value handling algorithm 
* Experiment review
  * Final best (slightly better than previous best)
  * Hard to determine whether the slightly score increase was due to the temporal widening of the dataset (selected year -> every year) or explicit missing value filling
  * The thing is that the validation score is hard to believe
---
### Submission 11 date : 2022-08-28 23:28:04
* Model : Sklearn's RandomForestRegressor
* Dataset
  * Dimension : 2D(tabular) | 63 columns
  * Consider t ~ t-6(unit time) | best among 6
  * Selected features (9)
    * Consider every feature of waterfall dataset
  * Temporal range : 2012, 2013, 2016, 2017, 2018, 2020, 2022 (Selected by features' variance)
    * Due to the memory limitation
  * Note
    * Fill missing every missing value as -9999
    * Remove instance when there is nan in y
* Target
  * Four classes of y for t+1
* Score
  * Validation score : around 2.7 (RMSE/R^2)
  * Public score : 4.0718 (RMSE/R^2)
* Strategy
  * Fill missing values with simple int(-9999) for using sklearn model
  * Simple 4-fold validation (test set ratio 0.25) for more reliable validation score
  * Sklearn's vanilla RF's parameters are intuitive and easy-to-understand
* Experiment review
  * Overfitting is suspected
  * K-fold validation seems not better than simple split
  * Validation strategy might need temporal consideration
  * Using all features might be a crucial reason for raising overfitting
---
### Submission 12 date : 2022-08-29 02:37:54
* Model : ExtraTreeRegressor
* Dataset
  * Dimension : 2D(tabular) | 63 columns
  * Consider t ~ t-6(unit time) | best among 6
  * Selected features (9)
    * Consider every feature of waterfall dataset
  * Temporal range : 2012, 2013, 2016, 2017, 2018, 2020, 2022 (Selected by features' variance)
    * Due to the memory limitation
  * Note
    * Fill missing every missing value as -9999
    * Remove instance when there is nan in y
* Target
  * Four classes of y for t+1
* Score
  * Validation score : around 2.43 (RMSE/R^2)
  * Public score : 3.9897 (RMSE/R^2)
* Strategy
  * Fill missing values with simple int(-9999) for using sklearn model
  * Simple 4-fold validation (test set ratio 0.25) for more reliable validation score
  * ExtraTreeRegressor is faster than vanilla RandomForest 
* Experiment review
  * Overfitting is suspected
  * K-fold validation seems not better than simple split AGAIN
  * Validation-Public score looks similar to the Sklearn's RandomForest
  * Validation strategy might need temporal consideration
  * Using all features might be a crucial reason for raising overfitting
  * Structural differences between tree-based models maybe not be the key. Probably the data has limited information to overcome the current score. Thus, extra data could be a solution
---
### Submission 13 date : 2022-09-01 01:43:31
* Model : Sklearn's RandomForestRegressor
* Dataset
  * Dimension : 2D(tabular) | 63 columns
  * Consider t ~ t-12(unit time) | best among 6, 12
  * Selected features (3)
    * 'fw_1018683', 'fw_1019630', 'fw_1018662' 
  * Temporal range : 2012, 2013, 2016, 2017, 2018, 2020, 2022 (Selected by features' variance)
    * Due to the memory limitation
  * Note
    * Fill missing every missing value as -9999
    * Remove instance when there is nan in y
* Target
  * Four classes of y for t+1
* Score
  * Validation score : around 3.76 (RMSE/R^2)
  * Public score : 3.5674 (RMSE/R^2)
* Strategy
  * Fill missing values with simple int(-9999) for using sklearn model
  * Simple 4-fold validation (test set ratio 0.25) for more reliable validation score
  * Simultaneous application of two efficient methods (RandomForest and k-fold validation) for overfitting
* Experiment review
  * It seems not overfit but both validation and public score is not good enough
  * The better public score than validation is good evidence of avoiding overfitting
  * Need to find a way for better score