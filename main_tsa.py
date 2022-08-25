import os
import tsa
import etl

folder = './grafici_tsa'
if not os.path.exists(folder):
    os.mkdir(folder)
data = etl.importa_tsa('./dataset/deaths.csv')
to_keep = 'Italy'
param = 'Smoking'
# 'ritorna dizionario con le crypto selezionate
df = etl.create_list(data, to_keep, param)
# 'Controlliamo se la serie è stazionaria con un ADFTest
df.index.freq = 'YS'
df = df.astype('float32')

tsa.basic_tsa(df, param, 'deaths')

tsa.adickeyfuller(df)

# 'Il p-value è di 0.66 per cui accettiamo l'ipotesi nulla e facciamo una differenziazione.

tsa.differentiation(df, param)

# 'La seconda diff va velocemente in negativo quindi prendiamo d uguale ad 1.

# 'Vediamo se c'è necessità di un contributo da parte della componente AR, ispezioniamo il grafico dell'AC parziale PACF.

#tsa.prediction_check(model_fit)
tsa.calcolo_ar(df, param)

# 'Il lag 1 è il più significativo poiché il numero di lag che sta fuori dal limite di significatività è pari a 1, quindi AR 1.
# 'Per il termine MA guardiamo l'autocorrelazione del primo ordine e vale sempre 1.

# '(p,d,q), p è il termine AR, d è il numero di diff necessarie, q è il termine MA.

model = tsa.arima(df, param)  # 'I risultati migliori ci sono con (1,1,0)

tsa.residui(model)

# 'Errori residui con media prossima allo 0.

tsa.actualvsfitted(model)

train = df[param][:24]
test = df[param][24:]
#train, test = train_test_split(df, test_size=0.15)

tsa.fc(train, test)
fc, act = tsa.sarimax(train, test)
results = tsa.forecast_accuracy(fc, act)
print(results)








