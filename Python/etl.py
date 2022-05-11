import os
import pandas as pd

df = pd.read_csv('./dataset/pokedex_21.csv')
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
to_drop_list =['pokedex_number', 'german_name', 'japanese_name']
df = df.drop(columns = to_drop_list)
print(df)



