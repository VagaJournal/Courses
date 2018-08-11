###### Data Preprocessing template ######
# Import required libraries
import pandas as pd
import numpy
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split

###### Load data into DataFrame ######
# Read the data into a DataFrame
movie_data_df = pd.read_csv('data/MovieStats.csv')

# pandas.DataFrame.head() -- Prints the first N rows of a DataFrame, 5 by Default
# pandas.DataFrame.tail() -- Prints the last N rows of a DataFrame, 5 by Default
print(movie_data_df.head())
## Output ##
#     Genre   Age   Budget Liked
# 0  Horror  23.0  57000.0   Yes
# 1  Comedy  34.0  77000.0   Yes
# 2  Action  27.0  56000.0    No
# 3  Comedy  42.0  34000.0   Yes
# 4  Action  32.0      NaN    No

###### Data slicing and Preprocessing ######
# Slice the dataset into Dependent and Independent values
ind_values = movie_data_df.iloc[:, :-1].values
dep_values = movie_data_df.iloc[:, 3].values

# Using Scikit-learn's Imputer to handle missing values
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(ind_values[:, 1:3])
ind_values[:, 1:3] = imputer.transform(ind_values[:, 1:3])
print(ind_values)
## Output ##
# array([['Horror', 23.0, 57000.0],
#        ['Comedy', 34.0, 77000.0],
#        ['Action', 27.0, 56000.0],
#        ['Comedy', 42.0, 34000.0],
#        ['Action', 32.0, 59666.666666666664],
#        ['Horror', 31.0, 65000.0],
#        ['Comedy', 31.22222222222222, 42000.0],
#        ['Horror', 30.0, 63000.0],
#        ['Action', 40.0, 76000.0],
#        ['Horror', 22.0, 67000.0]], dtype=object)

###### Encoding Categorical Data ######
# Encoding using Scikit-learn's LabelEncoder
# Indepent Variable Encoding
labelencoder_ind = LabelEncoder()
ind_values[:, 0] = labelencoder_ind.fit_transform(ind_values[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
ind_values = onehotencoder.fit_transform(ind_values).toarray()
print(ind_values)
## Output ##
# [0,  0,   1,  23,  57000],
# [0,  1,   0,  34,  77000],
# [1,  0,   0,  27,  56000],
# [0,  1,   0,  42,  34000],
# [1,  0,   0,  32,  59666],
# [0,  0,   1,  31,  65000],
# [0,  1,   0,  31,  42000],
# [0,  0,   1,  30,  63000],
# [1,  0,   0,  40,  76000],
# [0,  0,   1,  22,  67000]

# Dependent Variable Encoding
labelencoder_dep = LabelEncoder()
dep_values = labelencoder_dep.fit_transform(dep_values)
print(dep_values)
## Output ##
# [1,
#  1,
#  0,
#  1,
#  0,
#  0,
#  0,
#  1,
#  0,
#  1]

###### Splitting Data into Train/Test set ######
X_train, X_test, y_train, y_test = train_test_split(ind_values, dep_values,
                                                    test_size=0.2, random_state=0)
