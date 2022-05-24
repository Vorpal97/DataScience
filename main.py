import descriptive
import etl

df = etl.importa()

#descriptive.count_one_mode(df, "type_1", "Tipo", "Conteggio", "Conteggi per tipo", "/Users/Manuel/Desktop/grafico1.png")
#descriptive.count_one_mode(df, "generation", "Generazione", "Conteggio", "Conteggio per generazione", "/Users/Manuel/Desktop/grafico2.png")
#descriptive.count_one_mode(df, "status", "Status", "Conteggio", "status", "/Users/Manuel/Desktop/grafico3.png")
descriptive.inverted_count(df, "egg_type_1", "Conteggio", "Egg Type", "Conteggio per Egg Type", "/Users/Fiorenza/Desktop/grafico4.png")

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

descriptive.bar_of_pie(param_pie, param_bar)
