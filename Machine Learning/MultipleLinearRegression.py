###### Multi Linear Regression ######
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

###### Load data into DataFrame ######
# Read the data into a DataFrame
startups_data_df = pd.read_csv(
    'D:/Projects/Python_Projects/Courses/Machine Learning/data/StartupsData.csv')

# pandas.DataFrame.head() -- Prints the first N rows of a DataFrame, 5 by Default
# pandas.DataFrame.tail() -- Prints the last N rows of a DataFrame, 5 by Default
print(startups_data_df.head())
## Output ##
#   R&D Spend  Administration  Marketing Spend       State     Profit
# 0  165349.20       136897.80        471784.10    New York  192261.83
# 1  162597.70       151377.59        443898.53  California  191792.06
# 2  153441.51       101145.55        407934.54     Florida  191050.39
# 3  144372.41       118671.85        383199.62    New York  182901.99
# 4  142107.34        91391.77        366168.42     Florida  166187.94

###### Data slicing and Preprocessing ######
# Slice the dataset into Dependent and Independent values
ind_values = startups_data_df.iloc[:, :-1].values
dep_values = startups_data_df.iloc[:, 4].values

###### Encoding Categorical Data ######
# Encoding using Scikit-learn's LabelEncoder
# Indepent Variable Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
ind_values[:, 3] = labelencoder.fit_transform(ind_values[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
ind_values = onehotencoder.fit_transform(ind_values).toarray()

# Avoidintheg  Dummy Variable Trap
ind_values = ind_values[:, 1:]

###### Splitting Data into Train/Test set ######
from sklearn.cross_validation import train_test_split
ind_train, ind_test, dep_train, dep_test = train_test_split(ind_values, dep_values,
                                                            test_size=0.2, random_state=0)

###### Fit Linear Regression to the Training set ######
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(ind_train, dep_train)

###### Predicting the results using Test set ######
dep_pred = regressor.predict(ind_test)
print(dep_pred)
print(dep_test)

## Compare the Outputs ##
## dep_pred ##           ## dep_test ##
#  103015.20159796          103282.38
#  132582.27760816          144259.4
#  132447.73845175          146121.95
#  71976.09851259           77798.83
#  178537.48221054          191050.39
#  116161.24230163          105008.31
#  67851.69209676           81229.06
#  98791.73374688           97483.56
#  113969.43533012          110352.25
#  167921.0656955           166187.94
