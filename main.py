from matplotlib import pyplot as plt
import seaborn as sns
import descriptive
import classification
import etl
import clustering
import os
import numpy as np
folder = './grafici'
if not os.path.exists(folder):
    os.mkdir(folder)
df = etl.importa('./dataset/pokedex_21.csv')
dff = etl.importa_clf('./dataset/pokedex_21.csv')

####DESCRIPTIVE####
"""
#Primi grafici descrittivi, tutti conteggi
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


to_keep_list = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'catch_rate', 'height_m', 'weight_kg', 'base_experience', 'ability_1', 'ability_2', 'ability_hidden', 'base_friendship', 'total_points', 'growth_rate', 'status']
to_keep_list2 = ['against_normal', 'against_fire', 'against_water', 'against_electric','against_grass','against_ice','against_fight','against_poison','against_ground','against_flying','against_psychic','against_bug','against_rock','against_ghost','against_dragon','against_dark','against_steel','against_fairy']
to_keep_list3 = df.columns

df_pears = etl.filter_columns(df, to_keep_list3)
df_pears2 = etl.filter_columns(df, to_keep_list)
#correlazione senza valori
descriptive.pears_corr_plot(df_pears, 'Correlazione di Pearson (all attributes)', "./grafici/pears_novalues.png")
descriptive.pears_corr_plot(df_pears2, 'Correlazione di Pearson (some attributes)', "./grafici/pears_novalues2.png")
#correlazione con valori
#descriptive.pears_corr_wvalues(df_pears, 'Correlazione di Pearson', "./grafici/pears_values.png")

to_keep_list4 = ['status', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense']
df_clust = etl.filter_columns(df, to_keep_list)
df_clust2 = etl.filter_columns(df, to_keep_list4)

#descrizione veloce
#print(df_clust.describe())

descriptive.pairplot(df_clust, '1')
descriptive.pairplot(df_clust2, '2')
descriptive.heatmap(df_clust, '1')
descriptive.heatmap(df_clust2, '2')
descriptive.comparison_graph(df,'percentage_male', 'attack',50, 'attack by male gender probability')
descriptive.comparison_graph(df,'percentage_male', 'defense',50,'defense by male gender probability')
descriptive.comparison_graph(df,'percentage_male', 'hp',50,'hp by male gender probability')
descriptive.comparison_graph(df,'percentage_male', 'total_points',50, 'total points by male gender probability')

####CLUSTERING####
####KMEANS####

x = 'total_points'
y = 'attack'
data = df.iloc[:, [df.columns.get_loc(x), df.columns.get_loc(y)]].values
data = np.nan_to_num(data)
#metodo per capire il numero di cluster con cui inizializzare il kmeans
clustering.elbow_method(data)
clustering.kmeans_clustering(data, 3, 'kmeans_attack_&_total_points', x, y, 'Pokémon base', 'Pokémon forti/leggendari', 'Pokémon medio livello')
clustering.silhouette_kmeans(data, 3, 'Silhouette_KMeans_Atk&TotPoints')


x = 'total_points'
y = 'base_experience'
data = df.iloc[:, [df.columns.get_loc(x), df.columns.get_loc(y)]].values
#to remove rows with nan elements
data = data[~np.isnan(data).any(axis=1)]
#data = np.nan_to_num(data)
#metodo per capire il numero di cluster con cui inizializzare il kmeans
#clustering.elbow_method(data)
clustering.kmeans_clustering(data, 5, 'kmeans_baseExperience_&_total_points', x, y, 'Normali da livellare', 'Molto facili da livellare', 'Difficili da livellare', 'Facili da livellare', 'Molto difficili da livellare')
clustering.silhouette_kmeans(data, 5, 'Silhouette_KMeans_Exp&TotPoints')


####
x = 'total_points'
y = 'catch_rate'
data = df.iloc[:, [df.columns.get_loc(x), df.columns.get_loc(y)]].values
#to remove rows with nan elements
data = data[~np.isnan(data).any(axis=1)]
#data = np.nan_to_num(data)
#metodo per capire il numero di cluster con cui inizializzare il kmeans
#clustering.elbow_method(data)
clustering.kmeans_clustering(data, 5, 'kmeans_catchRate_&_total_points', x, y, 'Difficili da catturare', 'Normali da catturare', 'Molto difficili da catturare', 'Deboli e molto facili da catturare', 'Deboli e normali da catturare')
clustering.silhouette_kmeans(data, 5, 'Silhouette_KMeans_catch&TotPoints')
with open(folder + '/kmeans_catchRate_&_total_points_note.txt', 'w') as f:
    f.write('Notare come ci sono dei Pokémon difficilissimi da catturare (3%) con total points variabili e pokemon facilissimi da catturare con total points variabili. ')

####DBSCAN####

#PER CALCOLARE EPSILON USO IL K-NEAREST NEIGHBORS
x = 'total_points'
y = 'attack'
data = df.iloc[:, [df.columns.get_loc(x), df.columns.get_loc(y)]].values
data = np.nan_to_num(data)
#KNN
#clustering.nn(data)

clustering.dbscan(data, 'dbscan_attack_&_totalPoints', x, y)

####DBSCAN with PCA, 5 features####
x = 'total_points'
y = 'attack'
z = 'defense'
k = 'sp_attack'
y = 'sp_defense'
data = df.iloc[:, [df.columns.get_loc(x), df.columns.get_loc(y), df.columns.get_loc(z), df.columns.get_loc(k), df.columns.get_loc(y)]].values
data = data[~np.isnan(data).any(axis=1)]

data = clustering.pca(data)
clustering.dbscan(data, 'db_scan_5_features_PCA')
"""
###CLASSIFICAZIONE
classification.classification(dff)