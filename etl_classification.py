import pandas as pd


def importa(dataset):
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