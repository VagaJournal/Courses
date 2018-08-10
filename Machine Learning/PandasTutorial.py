###### Download the dataset from the link below ######
## https://www.kaggle.com/ramamet4/app-store-apple-data-set-10k-apps ##

# import the required packages
import pandas as pd

###### Load data into DataFrame ######
# Read the data into a DataFrame
app_store_df = pd.read_csv('D:/Projects/Python_Projects/Courses/Machine Learning/data/AppStoreData.csv')

# pandas.DataFrame.head() -- Prints the first N rows of a DataFrame, 5 by Default
# pandas.DataFrame.tail() -- Prints the last N rows of a DataFrame, 5 by Default
print(app_store_df.head())
## Output ##
#    Unnamed: 0         id                                         track_name   ...     ipadSc_urls.num lang.num  vpp_lic
# 0           1  281656475                                    PAC-MAN Premium   ...                   5       10        1
# 1           2  281796108                          Evernote - stay organized   ...                   5       23        1
# 2           3  281940292    WeatherBug - Local Weather, Radar, Maps, Alerts   ...                   5        3        1
# 3           4  282614216  eBay: Best App to Buy, Sell, Save! Online Shop...   ...                   5        9        1
# 4           5  282935706                                              Bible   ...                   5       45        1

# Use DataFrame.shape to check the how many rows and columns are in our DataFrame
print(app_store_df.shape)
## Output ##
# (7197, 17)

###### DataFrame Indexing ######
# The Dataframe.iloc method allows us to retrieve rows and columns by index
print(app_store_df.iloc[0:3, :])
## Output ##
#    Unnamed: 0         id                                         track_name   ...     ipadSc_urls.num lang.num  vpp_lic
# 0           1  281656475                                    PAC-MAN Premium   ...                   5       10        1
# 1           2  281796108                          Evernote - stay organized   ...                   5       23        1
# 2           3  281940292    WeatherBug - Local Weather, Radar, Maps, Alerts   ...                   5        3        1
# 3           4  282614216  eBay: Best App to Buy, Sell, Save! Online Shop...   ...                   5        9        1

# The Dataframe.loc method allows us to retrieve rows and columns by label
print(app_store_df.loc[:3, "track_name"])
## Output ##
# 0                                      PAC-MAN Premium
# 1                            Evernote - stay organized
# 2      WeatherBug - Local Weather, Radar, Maps, Alerts
# 3    eBay: Best App to Buy, Sell, Save! Online Shop...

# Specify more than one column by passing in a list
print(app_store_df.loc[:3,["track_name", "currency"]])
## Output ##
#                                           track_name currency
# 0                                    PAC-MAN Premium      USD
# 1                          Evernote - stay organized      USD
# 2    WeatherBug - Local Weather, Radar, Maps, Alerts      USD
# 3  eBay: Best App to Buy, Sell, Save! Online Shop...      USD

###### Series Objects ######
# Create a Series objects
series = pd.Series(["Apple","Google"])
# Print the series object to evaluate
print(series)
## Output ##
# 0     Apple
# 1     Google

# Access single column in an existing DataFrame with Series
print(app_store_df["track_name"])

# Access multiple columns by sending a list to DataFrame with Series
print(app_store_df[["track_name", "currency"]])