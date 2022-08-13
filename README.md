# han_river_water_level
Han River Water Level Prediction AI Competition for  Flood Safety of Paldang Dam hosted by Korea Hydro &amp; Nuclear Power Co., Ltd. at DACON

---
## Experiment log
* model : Sklearn's RandomForestRegressor
* Dataset
  * Dimension : 2D(tabular) | 44 columns
  * Consider t ~ t-21(unit time)
  * Selected features : 'fw_1018683', 'fw_1019630' because of its less missing point in test time range
* Target
  * Four classes of y for t+1
* Score
  * Validation score : around 4.27 (RMSE/R^2)
  * Public score : 4.219 (RMSE/R^2)
* Experiment review
  * The score is not low enough. However, it is hard to widen the past time range for giving the model more data.
