### Predicting Car Prices using K-Nearest Neighbors
# [Old Car Price data](https://archive.ics.uci.edu/ml/datasets/Automobile)
- The intention is to experiment different hyperparameters and variables to find the best performing (lowest RMSE) K-Nearest Neighbors

---

### Code and Resources Used
**Python Version:** 3.7\
**Packages:** pandas, numpy, sklearn, matplotlib, re

---

#### Data Cleaning:
- Columns were not labeled. Used the [Attribute Information](https://archive.ics.uci.edu/ml/datasets/Automobile) to extract column names using Regex, and inserted to column names.
- Replaced "?" values with np.nan
- Replaced car doors with numberical value
- Dropped all rows where target column ('price') is null.

#### Feature Engineering:
- For the purpose of making every datapoint have the same scale, and so each feature is equally important, we normalize all numerical columns with min-max normalization (x-min)/(max-min)

#### Initial Model Training:
- With initial value of k-neighbours of 5, K-Nearest Neighbors model is tried on all columns. Here is their RMSE performance: (y-axis: RMSE, x-axis: k)
![Each feature RMSE on k-value 5](/graphs/Diff%20k%20values%20vs%20each%20feature%20column.png?raw=true)

#### Model Optimization:
- With first model training, k-value of 5 seems to be showing the earliest low RMSE with feature 'curb-weight'.
- Next we will try different number of combinations of features from the top 5 performing features (curb-weight, highway-mpg, city-mpg, length, width)
- Result:
  - Lowest RMSE: Top 4 combined variables of k=5 had RMSE of 3022
  - Top 3 combined variables of k=5 had RMSE of 3226
  - Top 5 combined variables of k=5 had RMSE of 3367
  - Top 2 combined variable of k=5 had RMSE of 3460
  
#### Model Training with Multivariables and Hyperparameters
- After exploring 26 different k-values and combinations of top 3 to top 5 different variables, the lowest RMSE of 2646 was achieved by K-Nearest Neighbor of k=4 and combination of top 3 variables of curb-weight, highway-mp, and city-mpg
![Combined features RMSE on multiple k-values](/graphs/Diff%20k%20values%20vs%20diff%20combined%20feature%20columns.png?raw=true)
