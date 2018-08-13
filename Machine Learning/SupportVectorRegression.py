###### Support Vector Regression ######
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

###### Load data into DataFrame ######
# Read the data into a DataFrame
salaries_data_df = pd.read_csv(
    'D:/Projects/Python_Projects/Courses/Machine Learning/data/PositionSalariesData.csv')

# pandas.DataFrame.head() -- Prints the first N rows of a DataFrame, 5 by Default
# pandas.DataFrame.tail() -- Prints the last N rows of a DataFrame, 5 by Default
print(salaries_data_df.head())
## Output ##
#             Position  Level  Salary
# 0   Business Analyst      1   45000
# 1  Junior Consultant      2   50000
# 2  Senior Consultant      3   60000
# 3            Manager      4   80000
# 4    Country Manager      5  110000

###### Data slicing and Preprocessing ######
# Slice the dataset into Dependent and Independent values
ind_values = salaries_data_df.iloc[:, 1:2].values
dep_values = salaries_data_df.iloc[:, 2].values

###### Feature Scaling ######
sc_ind = StandardScaler()
sc_dep = StandardScaler()
ind_values = sc_ind.fit_transform(ind_values)
dep_values = sc_dep.fit_transform(dep_values.reshape(-1, 1))
print(ind_values)
print(dep_values)
## Output ##
## ind_values ##            ## dep_values ##
#  -1.5666989                  -0.72004253
#  -1.21854359                 -0.70243757
#  -0.87038828                 -0.66722767
#  -0.52223297                 -0.59680786
#  -0.17407766                 -0.49117815
#   0.17407766                 -0.35033854
#   0.52223297                 -0.17428902
#   0.87038828                  0.17781001
#   1.21854359                  0.88200808
#   1.5666989                   2.64250325

###### Fit SVR to the Dataset ######
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(ind_values,dep_values)

# Predicting a new result
dep_pred = regressor.predict(6.5)
dep_pred = sc_dep.inverse_transform(dep_pred)

# Visualising the SVR results
plt.scatter(ind_values, dep_values, color = 'blue')
plt.plot(ind_values, regressor.predict(ind_values), color = 'green')
plt.title('Salary Position (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(ind_values), max(ind_values), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(ind_values, dep_values, color = 'blue')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Salary Position (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()