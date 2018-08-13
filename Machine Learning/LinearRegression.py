###### Linear Regression ######
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

###### Load data into DataFrame ######
# Read the data into a DataFrame
vehicle_data_df = pd.read_csv(
    'D:/Projects/Python_Projects/Courses/Machine Learning/data/VehicleData.csv')

# pandas.DataFrame.head() -- Prints the first N rows of a DataFrame, 5 by Default
# pandas.DataFrame.tail() -- Prints the last N rows of a DataFrame, 5 by Default
print(vehicle_data_df.head())
## Output ##
#    Vehicle_age    Price
# 0          1.1  85645.0
# 1          1.3  84345.0
# 2          1.5  78949.0
# 3          2.0  74938.0
# 4          2.2  72403.0

###### Data slicing and Preprocessing ######
# Slice the dataset into Dependent and Independent values
ind_values = vehicle_data_df.iloc[:, :-1].values
dep_values = vehicle_data_df.iloc[:, 1].values

###### Splitting Data into Train/Test set ######
ind_train, ind_test, dep_train, dep_test = train_test_split(ind_values, dep_values,
                                                            test_size=0.2, random_state=0)

###### Fit Linear Regression to the Training set ######
regressor = LinearRegression()
regressor.fit(ind_train, dep_train)

###### Predicting the results using Test set ######
y_pred = regressor.predict(ind_test)

###### Visualising the Training set results ######
plt.scatter(ind_train, dep_train, color = 'blue')
plt.plot(ind_train, regressor.predict(ind_train), color = 'green')
plt.title('Price vs Age (Training set)')
plt.xlabel('Age of Vehicle')
plt.ylabel('Price')
plt.show()

###### Visualising the Test set results ######
plt.scatter(ind_test, dep_test, color = 'blue')
plt.plot(ind_train, regressor.predict(ind_train), color = 'green')
plt.title('Price vs Age (Test set)')
plt.xlabel('Age of Vehicle')
plt.ylabel('Price')
plt.show()
