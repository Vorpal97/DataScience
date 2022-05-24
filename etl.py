import os
import pandas as pd


def importa():
    df = pd.read_csv('./dataset/pokedex_21.csv')
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    to_drop_list =['pokedex_number', 'german_name', 'japanese_name']
    df = df.drop(columns = to_drop_list)
    return(df)


def filter_columns(df, to_keep):
    df2 = pd.DataFrame()
    for elem in to_keep:
        df2[elem] = df[elem]
    return df2



