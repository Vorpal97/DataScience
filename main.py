import descriptive
import etl

df = etl.importa()

descriptive.count_one_mode(df, "type_1", "Tipo", "Conteggio", "Conteggi per tipo", "./grafici/grafico1.png", False)
descriptive.count_one_mode(df, "generation", "Generazione", "Conteggio", "Conteggio per generazione", "./grafici/grafico2.png", False)
descriptive.count_one_mode(df, "status", "Status", "Conteggio", "status", "./grafici/grafico3.png", False)
descriptive.inverted_count(df, "egg_type_1", "Conteggio", "Egg Type", "Conteggio per Egg Type", "./grafici/grafico4.png")

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

descriptive.bar_of_pie(param_pie, param_bar, 'Percentuali tipologia', "./grafici/grafico5.png")

to_keep_list = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'catch_rate', 'height_m', 'weight_kg']
to_keep_list2 = ['against_normal','against_fire','against_water','against_electric','against_grass','against_ice','against_fight','against_poison','against_ground','against_flying','against_psychic','against_bug','against_rock','against_ghost','against_dragon','against_dark','against_steel','against_fairy']

df_pears = etl.filter_columns(df, to_keep_list)

#correlazione senza valori
descriptive.pears_corr_plot(df_pears, 'Correlazione di Pearson', "./grafici/grafico6.png")

#correlazione con valori
descriptive.pears_corr_wvalues(df_pears, 'Correlazione di Pearson', "./grafici/grafico7.png")
