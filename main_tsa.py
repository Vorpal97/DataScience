import os
import tsa
import etl



folder = './grafici_tsa'
if not os.path.exists(folder):
    os.mkdir(folder)
df = etl.importa_tsa('./dataset/merged_top_100_cry.csv')
coins_to_keep = ['bitcoin', 'ethereum', 'dogecoin', 'solana']
#ritorna dizionario con le crypto selezionate
df_list = etl.create_crypto_dict(df, coins_to_keep)

#aggiunge colonna avg
#df_list = etl.add_avg_column(df_list, coins_to_keep)


#Overview delle serie temporali
tsa.basic_tsa(df_list, 'bitcoin', 'Close', 'Value in $')
tsa.basic_tsa(df_list, 'ethereum', 'Close', 'Value in $')
tsa.basic_tsa(df_list, 'dogecoin', 'Close', 'Value in $')
tsa.basic_tsa(df_list, 'solana', 'Close', 'Value in $')

#studiare stazionalità serie, stima parametro d
tsa.adickeyfuller(df_list, 'bitcoin', 'Close')
tsa.adickeyfuller(df_list, 'ethereum', 'Close')
tsa.adickeyfuller(df_list, 'dogecoin', 'Close')
tsa.adickeyfuller(df_list, 'solana', 'Close')

#differenziazione della serie per calcolare parametro d
tsa.differentiation(df_list, 'bitcoin', 'Close')









