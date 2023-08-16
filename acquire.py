from env import get_db_url
import os
import pandas as pd


# 1. Make a function named get_titanic_data that returns the titanic data from the codeup data science database as a pandas data frame. 
# Obtain your data from the Codeup Data Science Database.

def get_titanic_data():
    """
    Gets all data from the titanic_db in sql. To make it work, use 'df = get_titanic_data()'

    arguments: none

    return: a pandas dataframe
    """
    filename = "titanic.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        query = "SELECT * FROM passengers"
        connection = get_db_url("titanic_db")
        df = pd.read_sql(query, connection)
        df.to_csv(filename, index=False)
    return df

# 2. Make a function named get_iris_data that returns the data from the iris_db on the codeup data science database as a pandas data frame. 
# The returned data frame should include the actual name of the species in addition to the species_ids. 
# Obtain your data from the Codeup Data Science Database.


def get_iris():
    """
    Gets all data from the iris_db in sql. To make it work, use 'df = get_iris()'

    arguments: none

    return: a pandas dataframe
    """
    filename = "iris.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        query = """
        SELECT DISTINCT *
        FROM measurements
        JOIN species
        USING (species_id);"""
        connection = get_db_url("iris_db")
        df = pd.read_sql(query, connection)
        df.to_csv(filename, index=False)
    return df

# 3. Make a function named get_telco_data that returns the data from the telco_churn database in SQL. 
# In your SQL, be sure to join contract_types, internet_service_types, payment_types tables with the customers table, 
# so that the resulting dataframe contains all the contract, payment, and internet service options. 
# Obtain your data from the Codeup Data Science Database.


def get_telco_data():
    """
    Gets all data from the telco_churn db in sql. To make it work, use 'df = get_telco_data()'

    arguments: none

    return: a pandas dataframe
    """
    filename = "telco.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        query = """
        SELECT *
        FROM customers
        JOIN contract_types
        USING (contract_type_id)
        JOIN internet_service_types
        USING (internet_service_type_id)
        JOIN payment_types
        USING (payment_type_id);"""
        connection = get_db_url("telco_churn")
        df = pd.read_sql(query, connection)
        df.to_csv(filename, index=False)
    return df

# 4. Once you've got your get_titanic_data, get_iris_data, and get_telco_data functions written, now it's time to add caching to them. 
# To do this, edit the beginning of the function to check for the local filename of telco.csv, titanic.csv, or iris.csv. 
# If they exist, use the .csv file. If the file doesn't exist, then produce the SQL and pandas necessary to create a dataframe, 
# then write the dataframe to a .csv file with the appropriate name.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Just added for my use

def get_summary(df):
    '''
    get_summary will take in one positional argument, a single pandas DF, 
    and will output info to the console regarding the following info:
    - print the first 3 rows
    - print the # of rows and columns
    - print the columns
    - print the dtypes of each col
    - print summary statistics
    
    return:none
    '''

    print('First 3 rows of the dataframe:')
    print(df.head(3))
    print('~~~~~~~~~~~~~~')
    print('Number of Rows and Cols in DF:')
    print(f'Rows: {df.shape[0]}, Cols: {df.shape[1]}')
    print('~~~~~~~~~~~~~~')
    print('Column Names:')
    [print(col) for col in df.columns]
    print('~~~~~~~~~~~~~~')
    [print(col,'- datatype:', df[col].dtype) for col in df.columns]
    print('~~~~~~~~~~~~~~')
    print(df.describe().T)
    print('~~~~~~~~~~~~~~')
    print('Descriptive stats for Object Variables: ')
    print(df.loc[:, df.dtypes=='O'].describe().T)
    print('~~~~~~~~~~~~~~')
    for col in df.loc[:, df.dtypes=='O']:
        if df[col].nunique() > 10:
            print(f'Column {col} has too many uniques ({df[col].nunique()}) to display')
        else:
            print(f' {col}: ', df[col].unique())

