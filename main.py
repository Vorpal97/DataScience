from matplotlib import pyplot as plt

import descriptive
import etl
import clustering
import os
import numpy as np
folder = './grafici'
if not os.path.exists(folder):
    os.mkdir(folder)
df = etl.importa()

#Primi 4 grafici descrittivi, tutti conteggi
mean1 = descriptive.count_kernel(df, 'weight_kg', 'blue', "./grafici/kernel_peso.png", mostra = True)
mean2 = descriptive.count_kernel(df, 'hp', 'green', "./grafici/kernel_hp.png", mostra = True)
mean3 = descriptive.count_kernel(df, 'generation', 'purple', "./grafici/per_generazione.png.png", mostra = True)
means = [mean1, mean2, mean3]
descriptive.count_one_mode(df, "type_1", "Tipo", "Conteggio", "Conteggi per tipo", "./grafici/per_tipo.png", True)
descriptive.count_one_mode(df, "status", "Status", "Conteggio", "status", "./grafici/per_status.png", True)
with open(folder + '/means.txt', 'w') as f:
    for elem in means:
        f.write(elem + '\n')

'''
#descriptive.count_one_mode(df, "generation", "Generazione", "Conteggio", "Conteggio per generazione", "./grafici/per_generazione.png", True)
descriptive.count_one_mode(df, "type_1", "Tipo", "Conteggio", "Conteggi per tipo", "./grafici/per_tipo.png", False)
descriptive.count_one_mode(df, "status", "Status", "Conteggio", "status", "./grafici/per_status.png", False)
descriptive.inverted_count(df, "egg_type_1", "Conteggio", "Egg Type", "Conteggio per Egg Type", "./grafici/per_uovo.png")
'''

#Calcola parametri per torta
s = df["status"].value_counts()
dict = s.to_dict()
count = list(dict.values())
pie_sum = 0
for elem in count:
    pie_sum = pie_sum + elem
pie_percentage = []
for elem in count:
    pie_percentage.append(round((elem/pie_sum),2))
max = max(pie_percentage)
pie = count
del pie[0]
bar_sum = 0
for elem in pie:
    bar_sum = bar_sum + elem
bar_percentage = []
for elem in pie:
    bar_percentage.append(round((elem/bar_sum),2))

min = 0
for elem in pie_percentage[1:]:
    min = min + elem

param_pie = {
    "ratios" : [min, max],
    "labels" : ["Legendary", "Normal"],
    "explode" : [0.1, 0]
}
param_bar = {
    "title" : "Legendary",
    "legend" : list(dict.keys())[1:],
    "ratios" : bar_percentage
}
#torta con barra
descriptive.bar_of_pie(param_pie, param_bar, 'Percentuali tipologia', "./grafici/torta.png")

to_keep_list = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'catch_rate', 'height_m', 'weight_kg']
to_keep_list2 = ['against_normal', 'against_fire', 'against_water', 'against_electric','against_grass','against_ice','against_fight','against_poison','against_ground','against_flying','against_psychic','against_bug','against_rock','against_ghost','against_dragon','against_dark','against_steel','against_fairy']
to_keep_list3 = df.columns
df_pears = etl.filter_columns(df, to_keep_list3)

#correlazione senza valori
descriptive.pears_corr_plot(df_pears, 'Correlazione di Pearson', "./grafici/pears_novalues.png")

#correlazione con valori
#descriptive.pears_corr_wvalues(df_pears, 'Correlazione di Pearson', "./grafici/pears_values.png")

df_clust = etl.filter_columns(df, ['catch_rate', 'weight_kg'])
df_clust = df_clust.dropna()
clustering.dbscan(df_clust, "./grafici/db_scan.png")

