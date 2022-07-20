import pandas as pd


def importa(dataset):
    df = pd.read_csv(dataset)
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    to_drop_list =['pokedex_number', 'german_name', 'japanese_name']
    df = df.drop(columns = to_drop_list)
    return(df)


def create_crypto_dict(df, coins):
    df_list = {}
    for elem in coins:
        df_list[elem] = (df.loc[df['Coin'] == elem])
    return df_list


def importa_tsa(dataset):
    df = pd.read_csv(dataset, index_col=None)
    return(df)


def filter_columns(df, to_keep):
    df2 = pd.DataFrame()
    for elem in to_keep:
        df2[elem] = df[elem]
    return df2



