import os

import pandas as pd
from sklearn.model_selection import train_test_split
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

<<<<<<< Updated upstream
print(df_list['bitcoin'].tail(600)['Close'])


# differenziazione della serie per calcolare parametro d
tsa.differentiation(df_list, 'bitcoin', 'Close')
tsa.calcolo_ar(df_list, 'bitcoin', 'Close') #dal grafico sembrava necessario scegliere come ma
# il valore 1, andando a vedere i risultati di ARIMA, è stato necessario toglierlo per ottenere
# un modello migliore
# Con AM e MA uguali 0, i p-value sono 0 ma aumenta l'AIC.
=======
# 'La seconda diff va velocemente in negativo quindi prendiamo d uguale ad 1.
>>>>>>> Stashed changes

# 'Vediamo se c'è necessità di un contributo da parte della componente AR, ispezioniamo il grafico dell'AC parziale PACF.

<<<<<<< Updated upstream
# plot dei residui
#residuals = tsa.residui(model_fit)
#tsa.prediction_check(model_fit)
=======
tsa.calcolo_ar(df, param)
>>>>>>> Stashed changes

# 'Il lag 1 è il più significativo poiché il numero di lag che sta fuori dal limite di significatività è pari a 1, quindi AR 1.
# 'Per il termine MA guardiamo l'autocorrelazione del primo ordine e vale sempre 1.


# '(p,d,q), p è il termine AR, d è il numero di diff necessarie, q è il termine MA.

model = tsa.arima(df, param)  # 'I risultati migliori ci sono con (1,1,0)

tsa.residui(model)

# 'Errori residui con media prossima allo 0.

tsa.actualvsfitted(model)


#df['Close'] = df.dropna()['Close'].tail(200)
#df = df.dropna(axis=1, how='all').tail(300)
#df.apply(lambda x: x.fillna(x.mean()),axis=0)
#df = df.tail(300)
#print(df.to_string())
#df = df.set_index([pd.Index(range(0, len(df)))])
#print(df)


train = df[param][:24]
test = df[param][24:]
#train, test = train_test_split(df, test_size=0.15)

tsa.fc(train, test)
fc, act = tsa.sarimax(train, test)
results = tsa.forecast_accuracy(fc, act)
print(results)








