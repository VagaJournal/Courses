###### Multi Linear Regression ######
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

###### Fit Linear Regression to the Training set ######
regressor = LinearRegression()
regressor.fit(ind_values, dep_values)

###### Fit Polynomial Regression to the dataset ######
poly_reg = PolynomialFeatures(degree=4)
ind_poly = poly_reg.fit_transform(ind_values)
poly_reg.fit(ind_poly, dep_values)
regressor_2 = LinearRegression()
regressor_2.fit(ind_poly, dep_values)

# Visualising the Polynomial Regression results
plt.scatter(ind_values, dep_values, color = 'blue')
plt.plot(ind_values, regressor_2.predict(poly_reg.fit_transform(ind_values)), color = 'green')
plt.title('Salary Decision (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(ind_values), max(ind_values), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(ind_values, dep_values, color = 'blue')
plt.plot(X_grid, regressor_2.predict(poly_reg.fit_transform(X_grid)), color = 'green')
plt.title('Salary Decision (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
regressor.predict(6.5)
## Output ##
# 330378.78787879

# Predicting a new result with Polynomial Regression
regressor_2.predict(poly_reg.fit_transform(6.5))
## Output ##
# 158862.45265155
