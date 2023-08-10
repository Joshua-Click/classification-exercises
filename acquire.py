import env
import os
import pandas as pd

# make a funcion that reads content in from sql
def grab_data(
    db,
    user=env.user,
    password=env.password,
    host=env.host):
    '''
    grab data will query data from a specified positional argument (string literal)
    schema from an assumed user, password, and host provided
    that they were imported from an env
    
    return: a pandas dataframe
    '''
    query = '''SELECT * FROM employees LIMIT 100'''
    connection = f'mysql+pymysql://{user}:{password}@{host}/{db}'
    df = pd.read_sql(query, connection)
    return df


def get_titanic_data():
    pass




def get_iris_data():
    pass




def get_telco_data():
    pass