import pandas as pd


def importa(dataset):
    df = pd.read_csv(dataset)
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    to_drop_list =['pokedex_number', 'german_name', 'japanese_name']
    df = df.drop(columns = to_drop_list)
    return(df)


def importa_clf(dataset):
    df = pd.read_csv(dataset)
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    to_drop_list =['pokedex_number', 'name', 'german_name', 'japanese_name', 'generation',
                   'status', 'species', 'type_number', 'type_2', 'height_m', 'weight_kg',
                   'abilities_number', 'ability_1', 'ability_2', 'ability_hidden', 'catch_rate', 'base_friendship', 'base_experience',
                   'growth_rate', 'egg_type_number', 'egg_type_1', 'egg_type_2', 'percentage_male', 'egg_cycles',
                   'against_normal', 'against_fire', 'against_water', 'against_electric',
                   'against_grass', 'against_ice', 'against_fight', 'against_poison',
                   'against_ground', 'against_flying', 'against_psychic', 'against_bug',
                   'against_rock', 'against_ghost', 'against_dragon', 'against_dark',
                   'against_steel', 'against_fairy']
    df = df.drop(columns = to_drop_list)
    return(df)


def create_crypto_dict(df, coins):
    df_list = {}
    for elem in coins:
        df_list[elem] = (df.loc[df['Coin'] == elem])
        #df_list[elem].reset_index(drop=True)
        df_list[elem] = df_list[elem].set_index('Date')
    return df_list


def create_list(data, keep, param):
    df = pd.DataFrame()
    #data['Date'] = pd.to_datetime(data['Date'])
    #data = data.set_index('Date')
    data = data.set_index('Year')
    data.index = pd.to_datetime(data.index, format='%Y')
    df = data.loc[data['Entity'] == keep]
    df = df[[param]]
    return df


def importa_tsa(dataset):
    df = pd.read_csv(dataset, sep=';', index_col=None)
    return(df)


def filter_columns(df, to_keep):
    df2 = pd.DataFrame()
    for elem in to_keep:
        df2[elem] = df[elem]
    return df2


def add_avg_column(df, coins_to_keep):
    for elem in coins_to_keep:
        df[elem]['Avg'] = (df[elem]['High'] + df[elem]['Low'])/2
    return df


